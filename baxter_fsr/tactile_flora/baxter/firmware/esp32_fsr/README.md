# esp32_fsr

Adafruit HUZZAH32 (ESP32 Feather, Digi-Key 1528-2514-ND) readout of the thin-film FSRs. It speaks
the same line protocol as `teensy_fsr` over USB serial, and it also streams that protocol over
WiFi UDP to `tactile_flora/baxter/udp_boards.py`. `tf baxter monitor`, `tf baxter calibrate` and
the ROS node work unchanged on either link.

SENSOR STATUS: the part is ASIN B0FGD267GK, a 4-pack of 0.4 mm thin-film sensors rated 20 g to
2 kg (0.2 to 19.6 N). The listing's photos show an RP-C10-ST-style round film. We have not
measured its resistance curve. `sensor_confirmed` stays false until R(F) is measured and
`r_ref_ohm` is chosen from it (step 2 below).

## Layout on Baxter: one board per gripper

Mount each Feather and its battery on the gripper body, after the last wrist joint (w2). The FSR
tails then run about 10 cm across the finger stroke and never cross a joint. Use
`SLOTS = {0, 1}` on the left gripper's board and `SLOTS = {2, 3}` on the right gripper's board.
The host merges the two boards into the four-channel stream. One board can also read all four
pads (`SLOTS = {0, 1, 2, 3}`, pins A2/A3/A4/32), but then the leads have to run along both arms
through every joint.

## Wiring (identical on every board)

Per pad: 3V3 -- FSR -- pin -- R_ref -- GND, plus 100 nF ceramic from the pin to GND, placed at
the Feather.

| Slot order on this board | Pin (silkscreen) | GPIO | ADC |
|---|---|---|---|
| first entry of SLOTS (finger a) | A2 | 34 | ADC1_CH6 |
| second entry (finger b) | A3 | 39 | ADC1_CH3 |
| third entry | A4 | 36 | ADC1_CH0 |
| fourth entry | 32 | 32 | ADC1_CH4 |
| rail monitor | 33 | 33 | ADC1_CH5: 3V3 -- 10 kOhm -- 33 -- 10 kOhm -- GND, 100 nF to GND |

- ADC1 pins only. ADC2 (A0, A1, A5, 12, 13, 14, 15, 27) stops working whenever WiFi is on.
- The rail monitor makes the counts ratiometric. The ESP32 ADC measures against an internal
  ~1.1 V reference, not 3V3, so without the monitor a sagging regulator shows up as a force
  change. On LiPo power the regulator drops out below roughly 3.4 V of battery. If you leave the
  monitor off, set `RAIL_PIN = -1` (a floating pin 33 reads garbage).
- The 100 nF cap holds the pin while the ADC's sample capacitor charges. With R_ref at about
  10 kOhm it is a ~1 ms low-pass, far above grasp bandwidth.
- FSR leads: do not put an iron on the printed silver traces (the PET melts). Use the crimped
  pins the films come with, or crimp/clincher connectors. Tape the tail for strain relief, and
  never fold it at the pins.

## Bring-up

1. Flash: Arduino IDE, board "Adafruit ESP32 Feather", or
   `arduino-cli compile --fqbn esp32:esp32:featheresp32 . && arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:featheresp32 .`
   (ESP32 core 3.3.x; this sketch compiles on 3.3.12). Set `BOARD_NAME`, `SLOTS`, `WIFI_SSID`
   and `WIFI_PASS` at the top first. The HUZZAH32's USB chip is a CP2104, so the port is
   `/dev/ttyUSB0`, and opening it resets the board (harmless: the boot text is rejected as bad
   lines).
2. Pick R_ref from measurement, not from the listing. With a multimeter on a bare film, read the
   resistance unloaded and then under 100 g, 500 g, 1 kg and 2 kg pressed through the puck you
   will mount (water on a kitchen scale works). Counts are `4095 * R_ref / (R_ref + R_fsr)`.
   Choose R_ref so that the largest grasp force you expect gives at most ~3000 counts (2.45 V,
   where the ESP32 ADC stops being linear; at 3.0 V the firmware pins the count to 4095 =
   saturated). Below ~190 counts (~0.15 V) the ESP32 reads 0, so light touch disappears if R_ref
   is too small. The best resolution comes when R_ref is near R_fsr at the force you care about
   most. 5.1 kOhm (the current `r_ref_ohm`, as built on the first board) is a starting guess
   only. Record what you pick in `configs/baxter.yaml`. Read the unloaded film on a range of at
   least 2 MOhm: on a low range an open film shows only the overrange mark. The calibration in
   step 5 measures the same thing on the mounted pads: it prints R_fsr at every push and the
   R_ref that would put your heaviest push at 3000 counts.
3. USB check: `tf baxter monitor --config configs/baxter.yaml --port /dev/ttyUSB0`. Press each
   pad and watch its count rise. In the Arduino Serial Plotter, send `P` to switch to plotter
   output and `P` again to switch back.
4. WiFi: run a dedicated 2.4 GHz network that the workstation is on. The simplest is the
   workstation's own hotspot:
   `nmcli device wifi hotspot ifname <wifi-if> ssid baxter-fsr password <pw> band bg`.
   Venue WiFi usually isolates clients and is congested. Open UDP 9870 if a firewall is on
   (`sudo ufw allow 9870/udp`). Each board broadcasts a 1 Hz beacon and the host answers `H`,
   so no IP addresses are configured anywhere.
   `tf baxter monitor --config configs/baxter.yaml` (serial_port `udp://0.0.0.0:9870`) prints
   one status line per board: name, slots, up/DOWN, gaps, vbat, 3v3 and read_us.
5. Calibrate after mounting, on the robot, with the scale rig below:
   `tf baxter calibrate --config configs/baxter.yaml --port /dev/ttyUSB0 --pads L_a,L_b`
   (pads are done one at a time in that order; `--pads R_a,R_b` for the right gripper, `--pads
   L_b` to redo one pad; without `--port` it uses `serial_port`). Redo a pad whenever its film
   is re-mounted, and all of them when R_ref changes.

## Calibration rig: one pad at a time against a kitchen scale

A kitchen scale stands on its edge, braced behind its base, and one pad at a time is pushed
against its platform (by closing the gripper or moving the arm). The scale then reads the force
through that pad. That holds only if:

- The puck is the pad's only contact. The puck and skin must stand proud of the finger face. If
  the finger face also touches the block or the platform, the pad sees less than the scale reads
  during calibration, and the fit then over-reports force in real grasps.
- Only the base touches the table and the brace. The reaction to the push has to go through the
  load cell: if the platform's edge rests on the table, friction there carries part of the push
  and the scale reads low. Shim under the base until a sheet of paper slides under the platform's
  edge, and brace the base (a wall or a heavy book behind it), not the platform.
- The push is square to the platform. A 10 degree tilt puts cos 10 = 0.985 of the force on the
  scale, a 1.5% error.
- The scale is tared in place: standing on its edge, braced, block attached, nothing pushing. On
  its side the platform's weight no longer loads the cell as designed and the tare removes that
  offset. Most kitchen scales re-zero when they power on, so after an auto-off, wake it with
  nothing pushing.
- The scale is set to grams. The script converts with g = 9.80665 m/s^2.

Procedure, for each pad in `--pads`:

1. Move the scale clear so nothing touches the pad, tare the scale, press enter: that pad's
   unloaded reading (0 N).
2. For each push: push the pad against the scale and hold. When the scale reading has settled
   (about 2 s), press enter and keep holding until "captured" prints (1 s of samples, median
   taken). Then type the scale reading in grams. `r` instead of a reading discards the push; use
   it whenever the script warns that the force moved during the capture.
3. Back off between pushes and work from light to heavy: 6 to 8 pushes from the lightest contact
   you care about to at least the hardest grasp of the demo, plus one or two repeats to see the
   repeatability. Keep the lightest push above about 20 g: kitchen scales track their zero and
   can swallow lighter loads.
4. Type `done`. It needs `calibration_min_points` (4) usable pushes; pushes that read nothing or
   saturate do not count. The pad's report prints and the file is saved before the next pad
   starts, so you can decide on R_ref after the first pad, and Ctrl-C later loses nothing already
   finished.

The script writes `configs/baxter_calibration.json`: a monotone piecewise-linear counts-to-newton
fit per pad, plus the raw points (`records`: grams, median counts, and dropped pushes with the
reason). Fits already in the file are kept if they were made with the same `r_ref_ohm` (a pad
redone replaces its own fit) and discarded otherwise. A pad with no fit publishes 0 N, and the
ROS node warns about it at start, so one calibrated gripper is enough to run.

Reading the report it prints per pad:

- Each push's grams, newtons, median counts and the film resistance R_fsr = R_ref * (4095 /
  counts - 1). That formula assumes ratiometric counts (rail monitor fitted, or USB power), and
  it is biased above 3000 counts.
- "reads nothing up to X g": below the ESP32's ~0.15 V floor. Contact lighter than X is
  invisible to this pad.
- "saturated from X g": the count hit 4095. R_ref is too large for that force.
- "forces above X g read as Y N": the fit clamps at the heaviest usable push, so push at least
  as hard as the hardest grasp you will record.
- "R_ref that puts X g at 3000 counts": computed from the heaviest push still in the linear
  range. When heavier pushes saturated or passed 3000 counts it is an upper bound. If you change
  the resistor, update `r_ref_ohm` and calibrate every pad again.

FSR films creep under a constant load and read differently loading and unloading (generic FSR
behavior, not yet measured on this film). Capture at a consistent time after the scale settles,
and always while pushing onto the scale, never while backing off.

## What to check on the first powered board

These are the numbers that would invalidate a design assumption. None has been measured yet:

- `read_us` (worst-case microseconds spent reading per 5 ms tick): must stay well under 5000. If
  it does not, lower `OVERSAMPLE`.
- `gaps` (samples lost over WiFi) and `dropped` (on-board queue overflow): both should stay at 0
  with the board 1 to 2 m from the AP.
- Noise: the std of each count with the pad unloaded and then under a steady 500 g, with WiFi
  on and with WiFi off (`WIFI_SSID = ""`). If WiFi raises it, check the rail monitor and the
  100 nF caps first.
- Battery: vbat over a 1 h session, to size the LiPo.
- The adc_cali line in the boot banner. "NONE (raw fallback)" means the chip has no eFuse ADC
  calibration and the counts are less linear (the empirical calibration still absorbs this).

## Protocol

USB serial at 921600 baud, the same lines as `teensy_fsr`: `<micros>,<a0>,<a1>,<a2>,<a3>\n` at
200 Hz, with slots this board does not own sent as 0. `Z\n` starts a 1 s baseline and the board
replies `Z,<b0>,<b1>,<b2>,<b3>\n`; unlike the Teensy, the stream keeps running during the
baseline. Lines starting with `#` are info lines (the host rejects them as bad lines). UDP
datagrams are a header `#B,<name>,<slots>,<boot>,<vbat_mv>,<rail_mv>,<dropped>,<read_us>` plus 4
sample lines (20 ms). The host stamps each sample on the board's clock, mapped to host time
through the lower envelope of arrival times, so WiFi jitter and stalls delay samples without
moving their stamps. Simulation (`tests/unit/test_baxter_udp_boards.py`: 2 ms minimum latency,
exponential jitter, 2% stalls of 50-150 ms) puts the stamps within +-3 ms of the true sample time
plus that minimum latency, and keeps two boards aligned within 3 ms. The real minimum latency
is not measured.
