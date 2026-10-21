# teensy_fsr

Teensy 4.0 readout of four thin-film force sensors (two per gripper).

SENSOR STATUS: the exact part (Amazon ASIN B0CZ6L5NMM) is NOT confirmed. Constants in the sketch
are FlexiForce A201 defaults (R_ref 4.7 kOhm) and each is marked `UNCONFIRMED`. `sensor_confirmed`
in `configs/baxter.yaml` stays false until the part is identified; the ROS node and `tf baxter
calibrate` print a warning while it is false.

## Wiring

Per channel: 3.3 V -- FSR -- analog pin -- R_ref (4.7 kOhm) -- GND. Pins A0..A3 map to
L_a, L_b, R_a, R_b (left gripper pads a/b, right gripper pads a/b). Do not feed 5 V to the pins.

## Flashing

Arduino IDE with Teensyduino: board "Teensy 4.0", open `teensy_fsr.ino`, Upload. Or
`arduino-cli compile --fqbn teensy:avr:teensy40 .` then `teensy_loader_cli`.

## Protocol (USB serial, 115200)

- Every 5 ms (200 Hz): `<micros>,<a0>,<a1>,<a2>,<a3>\n`. `micros` is the Teensy 32-bit
  microsecond counter (wraps every ~71 min; the host unwraps it). Counts are 12-bit (0..4095),
  16-sample oversampled, raw (no calibration in firmware).
- Host sends `Z\n`: the Teensy averages 1 s of readings (sample output pauses) and replies
  `Z,<b0>,<b1>,<b2>,<b3>\n`. Send it with the gripper open and nothing in it.

Host-side parsing lives in `tactile_flora/baxter/serial_protocol.py`.
