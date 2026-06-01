# Robot Nano firmware (Arduino Nano)

USB serial control of **head servos** (D9/D10) and **rotating base** (encoder N20 + TB6612 on D2–D7).

## Wiring — head servos

| Nano pin | Connect to |
|----------|------------|
| D9 | Pan servo signal |
| D10 | Tilt servo signal |
| GND | Servo GND, Pi GND (common ground) |

Power servos from an external **5 V** supply (not Nano 5V under load).

## Wiring — base motor (TB6612FNG + N20 encoder)

| Nano pin | Connects to | Purpose |
|----------|-------------|---------|
| **5V** | TB6612 VCC, STBY, motor Black | Logic + encoder 5 V |
| **GND** | TB6612 GND, motor Blue, 3.3 V supply (−) | Common ground |
| **D2** | Motor Yellow | Encoder A (interrupt) |
| **D3** | Motor Green | Encoder B (interrupt) |
| **D5** | TB6612 PWMA | Motor PWM |
| **D6** | TB6612 AIN1 | Direction 1 |
| **D7** | TB6612 AIN2 | Direction 2 |

| TB6612 pin | Connects to |
|------------|-------------|
| VM | 3.3 V motor supply (+) |
| AO1 / AO2 | Motor Red / White |

**Power checklist:** Do **not** connect 3.3 V VM to 5 V lines. Share one ground between Pi, Nano, driver, encoder, and motor supply.

## Calibration (required)

**Do not guess counts-per-degree from gear math.** Measure what your encoder actually reports.

### Step 1 — confirm encoder wiring

```bash
cd backend
python test_base_motor.py --watch
```

Turn the base **slowly**. You should feel the motor click as the encoder ticks.

- **POS** should climb into hundreds/thousands for a partial turn
- **A** and **B** should alternate between 0 and 1 (not stuck at 00 or 11 forever)

If POS stays near ±10: the **motor shaft** may not be turning (only the top plate), or D2/D3 wiring is wrong. Reflash firmware after code updates:

```bash
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:nano:cpu=atmega328 firmware/head_servo
```

### Step 2 — manual calibration

```bash
python test_base_motor.py --calibrate-manual --degrees 90 --write-config
```

1. Script zeros the encoder (`Z`)
2. Live **POS** display while you turn
3. Rotate exactly **90°** by hand (use a mark on the base)
4. Press **Enter**
5. Script computes `counts_per_degree = encoder_delta / 90`, applies `C…` to the Nano, optionally writes `config.yaml`

Repeat with `--degrees 180` if you want a second check.

### Step 3 — verify motor moves

```bash
python test_base_motor.py --zero --status
python test_base_motor.py --relative 30 --verify
```

Should move ~30° physically and print `OK B`.

Use `--apply-config-cpd` on later runs to load the saved value from config (motor moves also apply it automatically):

```bash
python test_base_motor.py --relative 30 --verify
```

### Mechanical reference (not used for config)

| Stage | Calculation | Counts / rev |
|-------|-------------|--------------|
| 7 PPR disk × quadrature | × 4 | 28 / motor rev |
| N20 gearbox | × 50:1 | 1,400 / D-shaft rev |
| External 11→110 gear | × 10 | ~14,000 / base rev |

Your measured value may differ (slip, wiring, partial quadrature). **Trust `--calibrate-manual`.**

## Upload

```bash
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328 firmware/head_servo
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:nano:cpu=atmega328 firmware/head_servo
```

Serial @ 115200 → `READY` on boot.

## Test scripts (from `backend/`)

Head servos:

```bash
python test_head_servos.py --verify --hold 5
```

Base motor:

```bash
python test_base_motor.py --watch
python test_base_motor.py --calibrate-manual --degrees 90 --write-config
python test_base_motor.py --apply-config-cpd --relative 30 --verify
python test_base_motor.py --absolute 0
python test_base_motor.py --status
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
| `?` | `POS <counts> DEG <float> CPD <float> BUSY 0\|1` |
| `I` | `ENC A=0\|1 B=0\|1 POS <counts>` (raw pin diagnostic) |
| `S` | Head pan bench sweep (~6 s) |

ACK lines (when `DEBUG_ACK 1`): `OK P85 T105`, `OK B45.0`, `OK C6.481`, `ERR B busy`, `ERR B timeout`.

## Pi config

```yaml
base:
  counts_per_degree: 1.0   # placeholder until --calibrate-manual --write-config
  move_timeout_sec: 15.0
```

## Troubleshooting

- **Port busy:** stop `robot_eyes` / other scripts using `/dev/ttyUSB0`.
- **POS stuck near ±10:** motor encoder not turning with the base, or A/B wires swapped/loose — use `--watch` and check A/B toggle; reflash firmware.
- **POS stuck at 0 when turning by hand:** encoder wiring (D2, D3, GND, 5 V).
- **ERR B busy:** wait for current move to finish before sending another B/M.
- **ERR B timeout:** run `--calibrate-manual` first; uncalibrated CPD causes runaway spin.
- **Servo issues:** see head-only tests with `test_head_servos.py --verify`.
