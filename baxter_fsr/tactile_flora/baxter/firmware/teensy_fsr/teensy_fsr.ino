// Teensy 4.0 FSR readout: 4 channels (L_a, L_b, R_a, R_b on A0..A3), 12-bit ADC,
// 16-sample oversampling, 200 Hz. Each tick prints "<micros>,<a0>,<a1>,<a2>,<a3>\n" at 115200.
// "Z\n" captures a 1 s baseline and echoes "Z,<b0>,<b1>,<b2>,<b3>\n". Raw counts only.
//
// Divider: 3.3 V -- FSR -- A_n -- R_REF -- GND (counts rise with force).

// UNCONFIRMED: set after sensor identity is confirmed (ASIN B0CZ6L5NMM; FlexiForce A201 default)
const float R_REF_OHM = 4700.0;  // documentation only; the host reads it from configs/baxter.yaml
// UNCONFIRMED: set after sensor identity is confirmed
const int ADC_BITS = 12;
// UNCONFIRMED: set after sensor identity is confirmed
const int OVERSAMPLE = 16;
// UNCONFIRMED: set after sensor identity is confirmed
const uint32_t TICK_US = 5000;  // 200 Hz loop

const int N_CH = 4;
const int PINS[N_CH] = {A0, A1, A2, A3};  // L_a, L_b, R_a, R_b
const uint32_t BASELINE_US = 1000000;

uint32_t next_tick_us = 0;

int readChannel(int pin) {
  uint32_t acc = 0;
  for (int i = 0; i < OVERSAMPLE; i++) acc += analogRead(pin);
  return (int)((acc + OVERSAMPLE / 2) / OVERSAMPLE);
}

void emitSample(uint32_t t_us) {
  Serial.print(t_us);
  for (int c = 0; c < N_CH; c++) {
    Serial.print(',');
    Serial.print(readChannel(PINS[c]));
  }
  Serial.print('\n');
}

void captureBaseline() {
  uint32_t t0 = micros();
  uint64_t sum[N_CH] = {0, 0, 0, 0};
  uint32_t n = 0;
  while ((uint32_t)(micros() - t0) < BASELINE_US) {
    for (int c = 0; c < N_CH; c++) sum[c] += readChannel(PINS[c]);
    n++;
  }
  Serial.print('Z');
  for (int c = 0; c < N_CH; c++) {
    Serial.print(',');
    Serial.print(n ? (int)(sum[c] / n) : 0);
  }
  Serial.print('\n');
  next_tick_us = micros() + TICK_US;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(ADC_BITS);
  analogReadAveraging(1);  // oversampling is done in software above
  next_tick_us = micros() + TICK_US;
}

void loop() {
  while (Serial.available() > 0) {
    int ch = Serial.read();
    if (ch == 'Z') captureBaseline();
  }
  uint32_t now = micros();
  if ((int32_t)(now - next_tick_us) >= 0) {
    emitSample(now);
    next_tick_us += TICK_US;
  }
}
