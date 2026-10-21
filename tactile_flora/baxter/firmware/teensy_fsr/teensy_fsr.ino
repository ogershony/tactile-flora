// Teensy 4.0 FSR readout: 4 channels (L_a, L_b, R_a, R_b on A0..A3), 12-bit ADC,
// 16-sample oversampling, 200 Hz, lines "<micros>,<a0>,<a1>,<a2>,<a3>\n" at 115200 baud.
// Raw counts only. "Z\n" captures and echoes a 1 s baseline. Stub until milestone 11.
const float R_REF_OHM = 4700.0;

void setup() {
  Serial.begin(115200);
}

void loop() {
}
