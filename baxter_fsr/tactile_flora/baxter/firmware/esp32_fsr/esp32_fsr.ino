// Adafruit HUZZAH32 (ESP32 Feather) FSR readout: up to 4 channels on ADC1 at 200 Hz, the same line
// protocol as teensy_fsr over USB serial, mirrored over WiFi UDP to tactile_flora/baxter/udp_boards.py.
// Each tick: "<micros>,<a0>,<a1>,<a2>,<a3>\n" in configs/baxter.yaml channel order; slots this board
// does not own are 0. "Z\n" (serial or UDP) captures a 1 s baseline and replies
// "Z,<b0>,<b1>,<b2>,<b3>\n" without pausing the stream. "P\n" on serial toggles Serial Plotter output.
//
// Divider per channel: 3V3 -- FSR -- pin -- R_ref -- GND, 100 nF from pin to GND (counts rise with
// force). Counts are ratiometric, 4095 * V_pin / V_3V3, with V_3V3 measured on RAIL_PIN, so they
// match a Teensy reading of the same divider and do not move with the regulator or the battery.
// ADC2 pins stop working while WiFi is on: ADC1 pins only.

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_adc/adc_cali_scheme.h"

// ---- per-board settings ------------------------------------------------------------------------
const char *BOARD_NAME = "left";  // short name shown by `tf baxter monitor`
// Channel slots this board fills: 0 L_a, 1 L_b, 2 R_a, 3 R_b. One board for all four: {0, 1, 2, 3}.
const int SLOTS[] = {0, 1};
const char *WIFI_SSID = "";  // empty: USB serial only
const char *WIFI_PASS = "";
// -------------------------------------------------------------------------------------------------

const int N_CH = 4;
const int N_OWN = sizeof(SLOTS) / sizeof(SLOTS[0]);
// The n-th entry of SLOTS is read on PINS[n]: silkscreen A2 (GPIO34), A3 (GPIO39), A4 (GPIO36), 32.
const int PINS[N_CH] = {34, 39, 36, 32};
const int RAIL_PIN = 33;  // 3V3 -- 10k -- 33 -- 10k -- GND; -1 if not fitted
const int VBAT_PIN = 35;  // A13, the on-board VBAT/2 divider
const float RAIL_NOMINAL_MV = 3300.0;  // used when RAIL_PIN is -1
const int OVERSAMPLE = 8;
const int OVERSAMPLE_RAIL = 4;  // the rail is low-passed below, so fewer reads
const float RAIL_ALPHA = 0.05;  // rail EMA per tick: ~100 ms time constant at 200 Hz
const uint32_t TICK_MS = 5;  // 200 Hz
const int ZERO_TICKS = 200;  // 1 s baseline
// The ESP32 ADC at 11 dB is linear to ~2.45 V and clips near 3.1 V; at or above SAT_MV the count is
// pinned to 4095 so the host's saturation check fires.
const uint32_t SAT_MV = 3000;
const uint16_t UDP_PORT = 9870;
const int BATCH = 4;  // samples per datagram (20 ms)
const uint32_t HOST_TIMEOUT_MS = 3000;  // stop streaming when the host stops sending "H"
const uint32_t SERIAL_BAUD = 921600;

struct Tick {
  uint32_t us;
  uint16_t c[N_CH];
  bool zero;  // true: c holds the baseline reply
};

QueueHandle_t ticks;
volatile bool zero_request = false;
volatile uint32_t rail_mv = 0, vbat_mv = 0, read_us = 0, dropped = 0;
bool cali_ok = true;

WiFiUDP udp;
bool udp_up = false;
IPAddress host_ip;
uint16_t host_port = 0;
uint32_t last_hello_ms = 0, last_beacon_ms = 0;
char boot_id[9];
char dgram[512];
int dgram_len = 0, dgram_n = 0;
bool plot = false;

uint32_t readMv(int pin, int n) {
  uint32_t acc = 0;
  for (int i = 0; i < n; i++) {
    // Without eFuse calibration analogReadMilliVolts returns 0; raw counts are a rough stand-in.
    acc += cali_ok ? analogReadMilliVolts(pin) : analogRead(pin) * 3100 / 4095;
  }
  return (acc + n / 2) / n;
}

void sendTick(const Tick &t) {
  if (xQueueSend(ticks, &t, 0) != pdTRUE) dropped = dropped + 1;
}

void sampler(void *) {
  float rail = RAIL_NOMINAL_MV;
  bool have_rail = false;
  uint32_t zero_sum[N_CH] = {0, 0, 0, 0};
  int zero_left = 0, n_tick = 0;
  uint32_t max_read_us = 0;
  TickType_t wake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(TICK_MS));
    uint32_t t0 = micros();
    if (RAIL_PIN >= 0) {
      float mv = 2.0f * readMv(RAIL_PIN, OVERSAMPLE_RAIL);
      rail = have_rail ? rail + RAIL_ALPHA * (mv - rail) : mv;
      have_rail = rail > 1000.0f;  // a missing divider reads ~0: keep re-seeding
      if (!have_rail) rail = RAIL_NOMINAL_MV;
    }
    Tick t = {};
    for (int n = 0; n < N_OWN; n++) {
      uint32_t mv = readMv(PINS[n], OVERSAMPLE);
      long c = lroundf(4095.0f * mv / rail);
      t.c[SLOTS[n]] = mv >= SAT_MV ? 4095 : (uint16_t)constrain(c, 0L, 4094L);
    }
    uint32_t t1 = micros();
    t.us = t0 + (t1 - t0) / 2;
    sendTick(t);
    max_read_us = max(max_read_us, t1 - t0);
    rail_mv = (uint32_t)rail;

    if (zero_request && zero_left == 0) {
      zero_request = false;
      zero_left = ZERO_TICKS;
      for (int s = 0; s < N_CH; s++) zero_sum[s] = 0;
    }
    if (zero_left > 0) {
      for (int n = 0; n < N_OWN; n++) zero_sum[SLOTS[n]] += t.c[SLOTS[n]];
      if (--zero_left == 0) {
        Tick z = {};
        z.zero = true;
        z.us = t.us;
        for (int s = 0; s < N_CH; s++) z.c[s] = (zero_sum[s] + ZERO_TICKS / 2) / ZERO_TICKS;
        sendTick(z);
      }
    }
    if (++n_tick >= ZERO_TICKS) {  // once a second
      n_tick = 0;
      vbat_mv = 2 * readMv(VBAT_PIN, OVERSAMPLE);
      read_us = max_read_us;
      max_read_us = 0;
    }
  }
}

bool streaming() {
  return udp_up && host_port != 0 && millis() - last_hello_ms < HOST_TIMEOUT_MS;
}

int writeHeader(char *buf, int cap) {
  char slots[N_CH + 1];
  for (int n = 0; n < N_OWN; n++) slots[n] = '0' + SLOTS[n];
  slots[N_OWN] = '\0';
  return snprintf(buf, cap, "#B,%s,%s,%s,%lu,%lu,%lu,%lu\n", BOARD_NAME, slots, boot_id,
                  (unsigned long)vbat_mv, (unsigned long)rail_mv, (unsigned long)dropped,
                  (unsigned long)read_us);
}

void flushDatagram() {
  if (dgram_len > 0 && streaming()) {
    udp.beginPacket(host_ip, host_port);
    udp.write((const uint8_t *)dgram, dgram_len);
    udp.endPacket();
  }
  dgram_len = dgram_n = 0;
}

void appendLine(const char *line, int n, bool flush_now) {
  if (!streaming()) return;
  if (dgram_len == 0) dgram_len = writeHeader(dgram, sizeof(dgram));
  if (dgram_len + n < (int)sizeof(dgram)) {
    memcpy(dgram + dgram_len, line, n);
    dgram_len += n;
    dgram_n++;
  }
  if (flush_now || dgram_n >= BATCH) flushDatagram();
}

void emit(const Tick &t) {
  char line[64];
  int n = t.zero ? snprintf(line, sizeof(line), "Z,%u,%u,%u,%u\n", t.c[0], t.c[1], t.c[2], t.c[3])
                 : snprintf(line, sizeof(line), "%lu,%u,%u,%u,%u\n", (unsigned long)t.us, t.c[0],
                            t.c[1], t.c[2], t.c[3]);
  appendLine(line, n, t.zero);
  static int n_plot = 0;
  if (!plot) {
    Serial.write((const uint8_t *)line, n);
  } else if (!t.zero && ++n_plot % 4 == 0) {  // 50 Hz is plenty for the plotter
    const char *names[N_CH] = {"L_a", "L_b", "R_a", "R_b"};
    for (int k = 0; k < N_OWN; k++) {
      Serial.printf("%s%s:%u", k ? "," : "", names[SLOTS[k]], t.c[SLOTS[k]]);
    }
    Serial.print('\n');
  }
}

void wifiTick() {
  if (!WIFI_SSID[0]) return;
  bool up = WiFi.status() == WL_CONNECTED;
  if (up && !udp_up) {
    udp_up = udp.begin(UDP_PORT);
    Serial.printf("# wifi up %s udp %u\n", WiFi.localIP().toString().c_str(), UDP_PORT);
  } else if (!up && udp_up) {
    udp.stop();
    udp_up = false;
    host_port = 0;
  }
  if (!udp_up) return;
  for (int len = udp.parsePacket(); len > 0; len = udp.parsePacket()) {
    char buf[32];
    int n = udp.read(buf, sizeof(buf));
    for (int i = 0; i < n; i++) {
      if (buf[i] == 'H') {
        host_ip = udp.remoteIP();
        host_port = udp.remotePort();
        last_hello_ms = millis();
      } else if (buf[i] == 'Z') {
        zero_request = true;
      }
    }
  }
  if (!streaming() && millis() - last_beacon_ms >= 1000) {  // discovery: header-only datagram
    last_beacon_ms = millis();
    char hdr[96];
    int n = writeHeader(hdr, sizeof(hdr));
    udp.beginPacket(WiFi.broadcastIP(), UDP_PORT);
    udp.write((const uint8_t *)hdr, n);
    udp.endPacket();
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);  // global: the millivolt calibration is built for this value
  adc_cali_line_fitting_efuse_val_t efuse;
  cali_ok = adc_cali_scheme_line_fitting_check_efuse(&efuse) == ESP_OK &&
            efuse != ADC_CALI_LINE_FITTING_EFUSE_VAL_DEFAULT_VREF;
  if (WIFI_SSID[0]) {
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // modem sleep adds ~100 ms latency spikes at each DTIM beacon
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
  snprintf(boot_id, sizeof(boot_id), "%08lx", (unsigned long)(esp_random() ^ micros()));
  ticks = xQueueCreate(256, sizeof(Tick));  // 1.28 s of samples
  xTaskCreatePinnedToCore(sampler, "fsr", 4096, nullptr, 5, nullptr, 1);
  Serial.printf("# esp32_fsr board=%s boot=%s adc_cali=%s\n", BOARD_NAME, boot_id,
                cali_ok ? "efuse" : "NONE (raw fallback)");
}

void loop() {
  while (Serial.available() > 0) {
    int ch = Serial.read();
    if (ch == 'Z') zero_request = true;
    if (ch == 'P') plot = !plot;
  }
  wifiTick();
  Tick t;
  if (xQueueReceive(ticks, &t, pdMS_TO_TICKS(2)) == pdTRUE) {
    do {
      emit(t);
    } while (xQueueReceive(ticks, &t, 0) == pdTRUE);
  }
}
