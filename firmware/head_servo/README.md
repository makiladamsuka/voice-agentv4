# Robot ESP32 firmware

USB serial control of **head servos** (PCA9685 over I2C) and **rotating base** (magnetic encoder + N20 + TB6612).

## Wiring — head servos (PCA9685)

| ESP32 pin | Connect to |
|-----------|------------|
| GPIO 21 | PCA9685 SDA |
| GPIO 22 | PCA9685 SCL |
| 3.3 V | PCA9685 VCC (logic) |
| GND | PCA9685 GND, Pi GND (common ground) |

| PCA9685 | Servo | Serial / Python |
|---------|--------|------------------|
| Channel 4 | **Pan** (horizontal) | `P` degrees, robottest **A/D**, face `norm_x` |
| Channel 5 | **Tilt** (vertical) | `T` degrees, robottest **W/S**, face `norm_y` |

| V+ | External **5 V** servo supply (not ESP32 5V) |
| GND | Servo GND (common with ESP32/Pi) |

See `backend/head_servo_axes.py` — must match `PAN_CH` / `TILT_CH` in `head_servo.ino` and `config.yaml` `servo.pan_ch` / `servo.tilt_ch`.

I2C address: **0x40** (default). Pulse range in firmware: **450–2600 µs** (matches `config.yaml` `pulse_min` / `pulse_max`).

## Wiring — ToF presence (TCA9548A + 3× VL53L0X)

Share the same I2C bus as the PCA9685 (**GPIO 21 SDA**, **GPIO 22 SCL**, 3.3 V, common GND).

| Device | Address | TCA9548A channel |
|--------|---------|------------------|
| TCA9548A mux | 0x70 | — |
| VL53L0X left | 0x29 | 0 |
| VL53L0X center | 0x29 | 1 |
| VL53L0X right | 0x29 | 2 |

One sensor per mux channel (all use 0x29; the mux isolates them). Add **4.7 kΩ** pull-ups on SDA/SCL if the bus is long or heavily loaded.

Boot log (after USB connect + reset):

- `FW head_servo+tof` then `READY` — ToF-capable firmware
- `TOF mux @ 0x70` (or `0x71`–`0x77` if ADDR pins strapped) then `TOF READY ch=0,1,2`
- If mux missing: `WARN TCA9548A not found` plus **`I2C scan:`** listing every device the ESP32 sees (helps debug wiring)

## Wiring — base motor (TB6612FNG + N20 encoder)

| ESP32 pin | Connects to | Purpose |
|-----------|-------------|---------|
| **GPIO 35** | Encoder C1 (Yellow) | Quadrature A (input-only) |
| **GPIO 34** | Encoder C2 (Green) | Quadrature B (input-only) |
| **GPIO 25** | TB6612 PWMA | Motor PWM |
| **GPIO 26** | TB6612 AIN1 | Direction 1 |
| **GPIO 27** | TB6612 AIN2 | Direction 2 |
| **GND** | TB6612 GND, motor Blue, encoder GND | Common ground |

GPIO **34** and **35** have no internal pull-ups. Use **10 kΩ pull-ups** on C1/C2 to 3.3 V (or encoder module built-in pull-ups).

| TB6612 pin | Notes |
|------------|--------|
| STBY | Tie **HIGH** (enable driver) |
| VM | Motor supply (+) per your motor voltage |
| VCC | 3.3 V logic |
| AO1 / AO2 | Motor outputs |

**Power checklist:** Share one ground between Pi, ESP32, driver, encoder, and motor supply. Do not tie motor VM to logic 5 V unless intended.

## Build and flash

Install toolchain once:

```bash
arduino-cli core update-index
arduino-cli core install esp32:esp32@2.0.17
arduino-cli lib install "Adafruit PWM Servo Driver Library" "Adafruit BusIO" "Adafruit_VL53L0X"
```

Compile and upload (stop `robot_eyes` / `robottest` first — one serial client):

Run from the **repo root** (`voice-agentv4/`), not `backend/`:

```bash
cd ~/Documents/voice-agentv4
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/head_servo
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/head_servo
```

Port may be `/dev/ttyACM0` on some boards. Pi `config.yaml` can set `servo.arduino_port` if auto-detect fails.

## Calibration (required)

**Do not guess counts-per-degree from gear math.** Measure what your encoder actually reports.

### Step 1 — confirm encoder wiring

```bash
cd backend
python tests/test_base_motor.py --watch
```

Turn the base **slowly**. **POS** should climb; **A** and **B** should toggle 0/1.

If POS stays near ±10: check C1/C2 wiring and **pull-ups** on GPIO 34/35, then reflash firmware.

### Step 2 — manual calibration

```bash
python tests/test_base_motor.py --calibrate-manual --degrees 90 --write-config
```

### Step 3 — verify motor moves

```bash
python tests/test_base_motor.py --zero --status
python tests/test_base_motor.py --relative 30 --verify
```

## Pi tests

```bash
cd backend
python tests/test_head_servos.py --verify
python tests/test_base_motor.py --watch
python tests/robottest.py
python tests/test_tof_sensors.py --port /dev/ttyUSB0
python tests/test_animations.py --list
python tests/tof_viz_server.py --port /dev/ttyUSB0
# Browser: http://<pi-ip>:8091/  (top-down radar, front + ±45°)
```

## Serial protocol

Newline-terminated lines:

| Command | Action |
|---------|--------|
| `P85.0 T105.0` | Head pan/tilt |
| `B+45.0` / `B-30.0` | Base relative degrees |
| `B45.0` | Base absolute degrees (zero at boot or `Z`) |
| `P85.0 T105.0 B+25.0` | Combined head + base |
| `M1000` / `M-1000` | Raw units; value ÷ 10 = encoder count delta |
| `C6.481` | Set counts per degree (runtime; until reboot) |
| `Z` | Zero encoder reference |
| `L` / `R` / `X` | Spin base left / right; `X` = stop |
| `?` | `POS <counts> DEG <float> CPD <float> BUSY 0\|1` |
| `H` | Handshake reply: `READY` |
| `I` | `ENC A=0\|1 B=0\|1 POS <counts>` |
| `S` | Pan bench sweep (~6 s) |
| `F` | One-shot ToF read → `TOF L=<mm> C=<mm> R=<mm> VALID=<bbb>` (`-1` = no target) |
| `O0` / `O1` | Disable / enable periodic ToF lines (default 5 Hz when enabled) |
| `O<hertz>` | Stream ToF at rate, e.g. `O5` |

ACK lines (when `DEBUG_ACK 1`): `OK P85 T105`, `OK B45.0`, `OK C6.481`, `ERR B busy`, `ERR B timeout`.

ToF example: `TOF L=842 C=-1 R=1205 VALID=101` (left/center valid, center out of range).

## Pi config

```yaml
servo:
  enabled: true
  backend: arduino   # USB to ESP32 (head + base on one link)
  arduino_port: ""
  arduino_baud: 115200
  pan_ch: 4          # firmware PCA9685 channels (reference only)
  tilt_ch: 5

base:
  counts_per_degree: 37.855556   # from --calibrate-manual --write-config
  move_timeout_sec: 15.0

tof:
  enabled: true
  poll_hz: 5
  present_max_mm: 1500
  absent_min_mm: 2000
```

Use `servo.backend: servokit` only if the PCA9685 is wired to the **Pi** I2C, not the ESP32.

## Troubleshooting

- **`Can't open sketch: firmware/head_servo`:** you ran `arduino-cli` from `backend/`; `cd` to the repo root first.
- **I2C scan shows `0x40` and `0x29` but no `0x70`:** PCA9685 and one VL53 are on the main bus; the **TCA9548A mux is missing or not powered** — all three ToF sensors must hang off mux channels 0/1/2, not SDA/SCL in parallel.
- **No READY on connect:** wrong port, baud, or firmware not flashed for ESP32.
- **Head ACK but servos still:** PCA9685 power, I2C address, channel wiring.
- **Base spin reversed:** swap AIN1/AIN2 in firmware or wiring.
- **ERR B timeout:** run manual calibration; bad CPD causes runaway.
