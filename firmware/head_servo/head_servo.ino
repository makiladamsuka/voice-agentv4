/*
 * Robot ESP32: head pan/tilt (PCA9685 I2C) + encoder base motor over USB serial.
 *
 * Head: PCA9685 @ 0x40, SDA=21 SCL=22, ch4 pan / ch5 tilt (external 5V servo power).
 * Base: GPIO35/34 encoder, GPIO25 PWM, GPIO26/27 -> TB6612FNG -> N20 motor.
 * CPD set at runtime via C command after manual calibration.
 *
 * Protocol (newline-terminated):
 *   P85.0 T105.0       head only
 *   A0=90.0 A1=45.0    arm servos (logical idx -> PCA ch 0,2,8,9)
 *   B+45.0 / B-30.0    base relative degrees
 *   B45.0              base absolute degrees (zero at boot or Z)
 *   P85.0 T105.0 B+25  combined
 *   M1000 / M-1000     raw units; value/10 = encoder counts delta
 *   C1.222             set counts per degree (runtime until reboot)
 *   Z                  zero encoder + angle reference
 *   L / R              spin base left / right (until X)
 *   X                  stop base motor (spin or encoder move)
 *   ?                  status: POS <n> DEG <f> CPD <f> BUSY 0|1
 *   H                  handshake (reply READY)
 *   V                  print all 6 servo home/stop positions (deg)
 *   I                  raw encoder pins: ENC A=0|1 B=0|1 POS <n>
 *   S                  pan servo sweep (bench)
 *   F                  one-shot ToF: TOF L=<mm> C=<mm> R=<mm> VALID=<bbb>
 *   O0 / O1            disable / enable ToF stream (default 5 Hz)
 *   O<hertz>           ToF stream rate (e.g. O5)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_VL53L0X.h>
#include <string.h>

#define DEBUG_ACK 1

// --- I2C / PCA9685 head servos ---
const int I2C_SDA_PIN = 21;
const int I2C_SCL_PIN = 22;
uint8_t tca9548Addr = 0x70;
const uint8_t VL53L0X_ADDR = 0x29;
const uint8_t TOF_MUX_CH[3] = {0, 1, 2};  // left, center, right
const uint8_t TOF_SENSOR_COUNT = 3;
const uint16_t TOF_INVALID_MM = 8190;
const uint16_t TOF_MIN_VALID_MM = 30;
const uint8_t TOF_MUX_SETTLE_MS = 40;
// Long-range profile: ~2 m on white targets (default mode tops out ~1.2 m).
const Adafruit_VL53L0X::VL53L0X_Sense_config_t TOF_SENSE_MODE =
    Adafruit_VL53L0X::VL53L0X_SENSE_LONG_RANGE;
// PAN_CH=4 pan servo, TILT_CH=5 tilt — must match config.yaml & backend/head_servo_axes.py
const uint8_t PAN_CH = 4;  // pan (horizontal)
const uint8_t TILT_CH = 5;  // tilt (vertical)

const float PAN_MIN = 40.0f;
const float PAN_MAX = 130.0f;
const float TILT_MIN = 80.0f;
const float TILT_MAX = 130.0f;
const float PAN_CENTER = 85.0f;
const float TILT_CENTER = 105.0f;
const int PULSE_MIN_US = 450;
const int PULSE_MAX_US = 2600;

// --- Base motor / encoder ---
const int ENC_A_PIN = 35;  // encoder C1 (input-only, external pull-up)
const int ENC_B_PIN = 34;  // encoder C2 (input-only, external pull-up)
const int MOTOR_PWM_PIN = 25;
const int MOTOR_AIN1_PIN = 26;
const int MOTOR_AIN2_PIN = 27;

const int LED_PIN = 2;
const int LINE_BUF_SIZE = 96;
const uint8_t ARM_CH_COUNT = 4;
// Botango export channels: 640->ch0, 642->ch2, 648->ch8, 649->ch9
const uint8_t ARM_CH[ARM_CH_COUNT] = {0, 2, 8, 9};
const float ARM_MIN = 0.0f;
const float ARM_MAX = 180.0f;
// Per-arm pulse limits (us) — match AnimationCommands.json rSVI2C min/max
const int ARM_PULSE_MIN_US[ARM_CH_COUNT] = {700, 1000, 1000, 925};
const int ARM_PULSE_MAX_US[ARM_CH_COUNT] = {2400, 2700, 1400, 1375};
const float ARM_HOME_DEG[ARM_CH_COUNT] = {0.0f, 180.0f, 90.0f, 90.0f};

// Boot placeholder — real CPD comes from manual calibration (C command).
float countsPerBaseDeg = 1.0f;
const int ENCODER_TOLERANCE = 50;
const int COARSE_ERR_COUNTS = 64;
const int COARSE_PWM = 155;
const int FINE_MIN_PWM = 85;
const int PWM_MAX = 180;
const int SPIN_PWM = 140;
const unsigned long STALL_MS = 4000;
const int MOTOR_DIR_SIGN = 1;
const int STALL_MIN_PROGRESS = 1;
const unsigned long MOVE_TIMEOUT_MS = 15000;

const float FINE_KP = 2.2f;

const int LEDC_FREQ_HZ = 20000;
const int LEDC_RES_BITS = 8;
const int LEDC_PWM_CHANNEL = 0;

Adafruit_PWMServoDriver pwm(0x40);
Adafruit_VL53L0X vl53[TOF_SENSOR_COUNT];
portMUX_TYPE encoderMux = portMUX_INITIALIZER_UNLOCKED;

bool tofMuxReady = false;
bool tofSensorOk[3] = {false, false, false};
int16_t tofDistanceMm[3] = {-1, -1, -1};
bool tofValid[3] = {false, false, false};
bool tofStreamEnabled = false;
float tofStreamHz = 5.0f;
unsigned long lastTofSweepMs = 0;
unsigned long lastTofStreamMs = 0;

struct ParsedCommand {
  bool hasPan;
  bool hasTilt;
  bool hasBase;
  bool hasRaw;
  bool hasCpd;
  bool hasArm;
  bool baseRelative;
  float pan;
  float tilt;
  float baseDeg;
  long rawUnits;
  float cpdValue;
  float armDeg[ARM_CH_COUNT];
  bool armSet[ARM_CH_COUNT];
};

float panAngle = PAN_CENTER;
float tiltAngle = TILT_CENTER;
float armAngles[ARM_CH_COUNT] = {
  ARM_HOME_DEG[0], ARM_HOME_DEG[1], ARM_HOME_DEG[2], ARM_HOME_DEG[3]
};
unsigned long lastCommandMs = 0;

char lineBuffer[LINE_BUF_SIZE];
uint8_t lineLen = 0;

volatile long encoderCount = 0;
uint8_t encLastState = 0;

const int8_t ENC_QUAD_TABLE[16] = {
  0, 1, -1, 0,
  -1, 0, 0, 1,
  1, 0, 0, -1,
  0, -1, 1, 0
};

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

int mapAngleToUs(float deg, float degMin, float degMax, int pulseMinUs, int pulseMaxUs) {
  float span = degMax - degMin;
  if (span < 0.001f) return pulseMinUs;
  float t = (deg - degMin) / span;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return pulseMinUs + (int)(t * (float)(pulseMaxUs - pulseMinUs));
}

void setServoPulseUs(uint8_t ch, int pulseUs) {
  pulseUs = constrain(pulseUs, PULSE_MIN_US, PULSE_MAX_US);
  uint32_t tick = ((uint32_t)pulseUs * 4096UL) / 20000UL;
  if (tick >= 4096) {
    tick = 4095;
  }
  pwm.setPWM(ch, 0, tick);
}

long readEncoderCount() {
  portENTER_CRITICAL(&encoderMux);
  long v = encoderCount;
  portEXIT_CRITICAL(&encoderMux);
  return v;
}

void serviceEncoder() {
  uint8_t a = digitalRead(ENC_A_PIN) & 1;
  uint8_t b = digitalRead(ENC_B_PIN) & 1;
  uint8_t state = (a << 1) | b;
  if (state == encLastState) {
    return;
  }
  uint8_t idx = (encLastState << 2) | state;
  int8_t delta = ENC_QUAD_TABLE[idx];
  if (delta != 0) {
    encoderCount += delta;
    encLastState = state;
  }
}

void serviceEncoderLocked() {
  portENTER_CRITICAL(&encoderMux);
  serviceEncoder();
  portEXIT_CRITICAL(&encoderMux);
}

void syncEncoderState() {
  uint8_t a = digitalRead(ENC_A_PIN) & 1;
  uint8_t b = digitalRead(ENC_B_PIN) & 1;
  encLastState = (a << 1) | b;
}

long zeroOffset = 0;
long moveTargetCount = 0;
long moveStartCount = 0;
bool baseBusy = false;
bool moveActive = false;
unsigned long moveStartMs = 0;
unsigned long lastProgressMs = 0;
long lastProgressCount = 0;
long lastErrorAbs = 0;
float ackBaseDeg = 0.0f;
bool pendingBaseAck = false;
int spinPwm = 0;

float countsToDeg(long counts) {
  return (float)(counts - zeroOffset) / countsPerBaseDeg;
}

long degToCounts(float deg) {
  return zeroOffset + (long)(deg * countsPerBaseDeg);
}

bool setCountsPerBaseDeg(float cpd) {
  if (cpd < 0.05f || cpd > 200.0f) {
    return false;
  }
  countsPerBaseDeg = cpd;
  return true;
}

void motorPwmWrite(int magnitude) {
  magnitude = constrain(magnitude, 0, PWM_MAX);
  int duty = (magnitude * 255) / PWM_MAX;
  ledcWrite(LEDC_PWM_CHANNEL, duty);
}

void motorStop() {
  ledcWrite(LEDC_PWM_CHANNEL, 0);
  digitalWrite(MOTOR_AIN1_PIN, LOW);
  digitalWrite(MOTOR_AIN2_PIN, LOW);
}

void motorDrive(int pwm) {
  pwm = constrain(pwm * MOTOR_DIR_SIGN, -PWM_MAX, PWM_MAX);
  if (pwm == 0) {
    motorStop();
    return;
  }
  if (pwm > 0) {
    digitalWrite(MOTOR_AIN1_PIN, HIGH);
    digitalWrite(MOTOR_AIN2_PIN, LOW);
    motorPwmWrite(pwm);
  } else {
    digitalWrite(MOTOR_AIN1_PIN, LOW);
    digitalWrite(MOTOR_AIN2_PIN, HIGH);
    motorPwmWrite(-pwm);
  }
}

void stopBaseMotion() {
  spinPwm = 0;
  moveActive = false;
  baseBusy = false;
  motorStop();
}

void startBaseSpin(int pwm) {
  moveActive = false;
  pendingBaseAck = false;
  spinPwm = constrain(pwm, -SPIN_PWM, SPIN_PWM);
  baseBusy = spinPwm != 0;
}

void startBaseSpinLeft() {
  startBaseSpin(-SPIN_PWM);  // increases POS
}

void startBaseSpinRight() {
  startBaseSpin(SPIN_PWM);   // decreases POS
}

void printServoAck(float pan, float tilt) {
#if DEBUG_ACK
  Serial.print(F("OK P"));
  Serial.print((int)round(pan));
  Serial.print(F(" T"));
  Serial.println((int)round(tilt));
#endif
}

void printBaseAck(float deg) {
#if DEBUG_ACK
  Serial.print(F("OK B"));
  Serial.println(deg, 1);
#endif
}

void printBaseBusy() {
  Serial.println(F("ERR B busy"));
}

void writeAngles(float pan, float tilt, bool emitAck) {
  panAngle = pan;
  tiltAngle = tilt;
  Wire.beginTransmission(0x40);
  if (Wire.endTransmission() != 0) {
    if (emitAck) {
      Serial.println(F("ERR PCA9685"));
    }
    return;
  }
  setServoPulseUs(PAN_CH, mapAngleToUs(pan, PAN_MIN, PAN_MAX, PULSE_MIN_US, PULSE_MAX_US));
  setServoPulseUs(TILT_CH, mapAngleToUs(tilt, TILT_MIN, TILT_MAX, PULSE_MIN_US, PULSE_MAX_US));
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(LED_PIN, LOW);
  if (emitAck) printServoAck(pan, tilt);
}

void writeArmAngles(const ParsedCommand &cmd) {
  for (uint8_t i = 0; i < ARM_CH_COUNT; i++) {
    if (!cmd.armSet[i]) continue;
    float a = clampf(cmd.armDeg[i], ARM_MIN, ARM_MAX);
    armAngles[i] = a;
    setServoPulseUs(
      ARM_CH[i],
      mapAngleToUs(a, ARM_MIN, ARM_MAX, ARM_PULSE_MIN_US[i], ARM_PULSE_MAX_US[i])
    );
  }
}

bool startBaseMoveToCount(long targetCount, float ackDeg) {
  if (baseBusy) {
    printBaseBusy();
    return false;
  }
  moveTargetCount = targetCount;
  ackBaseDeg = ackDeg;
  moveStartCount = readEncoderCount();
  lastProgressCount = moveStartCount;
  lastErrorAbs = labs(targetCount - moveStartCount);
  moveActive = true;
  baseBusy = true;
  moveStartMs = millis();
  lastProgressMs = moveStartMs;
  return true;
}

bool startBaseRelativeDeg(float deltaDeg) {
  long pos = readEncoderCount();
  long deltaCounts = (long)(deltaDeg * countsPerBaseDeg);
  long target = pos + deltaCounts;
  return startBaseMoveToCount(target, deltaDeg);
}

bool startBaseAbsoluteDeg(float deg) {
  long target = degToCounts(deg);
  return startBaseMoveToCount(target, deg);
}

bool startBaseRawUnits(long units) {
  long pos = readEncoderCount();
  long deltaCounts = units / 10;
  long target = pos + deltaCounts;
  float ackDeg = (countsPerBaseDeg > 0.05f)
    ? (float)deltaCounts / countsPerBaseDeg
    : (float)deltaCounts;
  return startBaseMoveToCount(target, ackDeg);
}

void zeroBaseReference() {
  if (baseBusy) {
    printBaseBusy();
    return;
  }
  portENTER_CRITICAL(&encoderMux);
  encoderCount = 0;
  zeroOffset = 0;
  syncEncoderState();
  portEXIT_CRITICAL(&encoderMux);
  moveTargetCount = 0;
  ackBaseDeg = 0.0f;
  Serial.println(F("OK Z"));
}

void printEncoderPins() {
  serviceEncoderLocked();
  uint8_t a = digitalRead(ENC_A_PIN) & 1;
  uint8_t b = digitalRead(ENC_B_PIN) & 1;
  long pos = readEncoderCount();
  Serial.print(F("ENC A="));
  Serial.print(a);
  Serial.print(F(" B="));
  Serial.print(b);
  Serial.print(F(" POS "));
  Serial.println(pos);
}

void printServoStopPose() {
  Serial.print(F("STOP P"));
  Serial.print(PAN_CENTER, 1);
  Serial.print(F(" T"));
  Serial.print(TILT_CENTER, 1);
  Serial.print(F(" ch0="));
  Serial.print(ARM_HOME_DEG[0], 1);
  Serial.print(F(" ch2="));
  Serial.print(ARM_HOME_DEG[1], 1);
  Serial.print(F(" ch8="));
  Serial.print(ARM_HOME_DEG[2], 1);
  Serial.print(F(" ch9="));
  Serial.println(ARM_HOME_DEG[3], 1);
}

void printStatus() {
  long pos = readEncoderCount();
  Serial.print(F("POS "));
  Serial.print(pos);
  Serial.print(F(" DEG "));
  Serial.print(countsToDeg(pos), 2);
  Serial.print(F(" CPD "));
  Serial.print(countsPerBaseDeg, 3);
  Serial.print(F(" BUSY "));
  Serial.println(baseBusy ? 1 : 0);
}

void parseCommandLine(const char *line, ParsedCommand &cmd) {
  cmd.hasPan = false;
  cmd.hasTilt = false;
  cmd.hasBase = false;
  cmd.hasRaw = false;
  cmd.hasCpd = false;
  cmd.hasArm = false;
  cmd.baseRelative = false;
  cmd.pan = PAN_CENTER;
  cmd.tilt = TILT_CENTER;
  cmd.baseDeg = 0.0f;
  cmd.rawUnits = 0;
  cmd.cpdValue = 0.0f;
  for (uint8_t i = 0; i < ARM_CH_COUNT; i++) {
    cmd.armDeg[i] = armAngles[i];
    cmd.armSet[i] = false;
  }

  char buf[LINE_BUF_SIZE];
  strncpy(buf, line, LINE_BUF_SIZE - 1);
  buf[LINE_BUF_SIZE - 1] = '\0';

  char *token = strtok(buf, " ");
  while (token != NULL) {
    char c = token[0];
    if (c == 'P' || c == 'p') {
      cmd.hasPan = true;
      cmd.pan = atof(token + 1);
    } else if (c == 'T' || c == 't') {
      cmd.hasTilt = true;
      cmd.tilt = atof(token + 1);
    } else if (c == 'B' || c == 'b') {
      cmd.hasBase = true;
      if (token[1] == '+' || token[1] == '-') {
        cmd.baseRelative = true;
        cmd.baseDeg = atof(token + 1);
      } else {
        cmd.baseRelative = false;
        cmd.baseDeg = atof(token + 1);
      }
    } else if (c == 'M' || c == 'm') {
      cmd.hasRaw = true;
      cmd.rawUnits = atol(token + 1);
    } else if (c == 'C' || c == 'c') {
      cmd.hasCpd = true;
      cmd.cpdValue = atof(token + 1);
    } else if (c == 'A' || c == 'a') {
      // A<idx>=<deg>, e.g. A0=90
      char *eq = strchr(token, '=');
      if (eq != NULL && eq > token + 1) {
        int idx = atoi(token + 1);
        if (idx >= 0 && idx < ARM_CH_COUNT) {
          cmd.hasArm = true;
          cmd.armSet[idx] = true;
          cmd.armDeg[idx] = atof(eq + 1);
        }
      }
    }
    token = strtok(NULL, " ");
  }
}

void runPanSweep() {
  const unsigned long stepMs = 100;
  const int steps = 30;
  for (int i = 0; i <= steps; i++) {
    float pan = PAN_MIN + (PAN_MAX - PAN_MIN) * ((float)i / (float)steps);
    writeAngles(pan, TILT_CENTER, true);
    delay(stepMs);
  }
  for (int i = 0; i <= steps; i++) {
    float pan = PAN_MAX - (PAN_MAX - PAN_CENTER) * ((float)i / (float)steps);
    writeAngles(pan, TILT_CENTER, true);
    delay(stepMs);
  }
  lastCommandMs = millis();
}

void applyParsedCommand(const ParsedCommand &cmd) {
  if (cmd.hasCpd) {
    if (setCountsPerBaseDeg(cmd.cpdValue)) {
      Serial.print(F("OK C"));
      Serial.println(countsPerBaseDeg, 3);
    } else {
      Serial.println(F("ERR C range"));
    }
  }
  if (cmd.hasPan || cmd.hasTilt) {
    float p = cmd.hasPan ? clampf(cmd.pan, PAN_MIN, PAN_MAX) : panAngle;
    float t = cmd.hasTilt ? clampf(cmd.tilt, TILT_MIN, TILT_MAX) : tiltAngle;
    writeAngles(p, t, true);
    lastCommandMs = millis();
  }
  if (cmd.hasArm) {
    writeArmAngles(cmd);
  }
  if (cmd.hasRaw) {
    startBaseRawUnits(cmd.rawUnits);
  } else if (cmd.hasBase) {
    if (cmd.baseRelative) {
      startBaseRelativeDeg(cmd.baseDeg);
    } else {
      startBaseAbsoluteDeg(cmd.baseDeg);
    }
  }
}

int16_t sanitizeTofRangeMm(uint16_t raw) {
  if (raw >= TOF_INVALID_MM) return -1;
  return (int16_t)raw;
}

void printTofLine() {
  Serial.print(F("TOF L="));
  Serial.print(tofDistanceMm[0]);
  Serial.print(F(" C="));
  Serial.print(tofDistanceMm[1]);
  Serial.print(F(" R="));
  Serial.print(tofDistanceMm[2]);
  Serial.print(F(" VALID="));
  Serial.print(tofValid[0] ? '1' : '0');
  Serial.print(tofValid[1] ? '1' : '0');
  Serial.println(tofValid[2] ? '1' : '0');
}

bool tcaSelect(uint8_t channel) {
  if (channel > 7 || tca9548Addr == 0) return false;
  Wire.beginTransmission(tca9548Addr);
  Wire.write((uint8_t)(1 << channel));
  return Wire.endTransmission() == 0;
}

void printI2cScan() {
  Serial.println(F("I2C scan:"));
  uint8_t count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  dev 0x%02X\n", addr);
      count++;
    }
  }
  if (count == 0) {
    Serial.println(F("  (no devices — check SDA/SCL power)"));
  }
}

bool findTca9548Addr() {
  for (uint8_t addr = 0x70; addr <= 0x77; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      tca9548Addr = addr;
      Serial.printf("TOF mux @ 0x%02X\n", addr);
      return true;
    }
  }
  tca9548Addr = 0;
  return false;
}

bool initVl53OnBus(Adafruit_VL53L0X &sensor) {
  if (!sensor.begin(VL53L0X_ADDR, false, &Wire, TOF_SENSE_MODE)) {
    return false;
  }
  sensor.startRangeContinuous();
  return true;
}

bool recoverVl53OnMux(uint8_t idx) {
  if (!tofMuxReady || idx >= TOF_SENSOR_COUNT) {
    return false;
  }
  if (!tcaSelect(TOF_MUX_CH[idx])) {
    return false;
  }
  delay(TOF_MUX_SETTLE_MS);
  if (!initVl53OnBus(vl53[idx])) {
    tofSensorOk[idx] = false;
    return false;
  }
  tofSensorOk[idx] = true;
  return true;
}

void printTofMuxScan() {
  if (tca9548Addr == 0) {
    Serial.println(F("TOF mux scan: mux not found"));
    return;
  }
  Serial.println(F("TOF mux scan:"));
  Adafruit_VL53L0X tempVl53;
  uint8_t found = 0;
  for (uint8_t ch = 0; ch < 8; ch++) {
    if (!tcaSelect(ch)) {
      continue;
    }
    delay(TOF_MUX_SETTLE_MS);
    if (tempVl53.begin(VL53L0X_ADDR, false, &Wire)) {
      Serial.print(F("  ch "));
      Serial.print(ch);
      Serial.println(F(" VL53L0X"));
      found++;
    }
  }
  if (found == 0) {
    Serial.println(F("  (no VL53L0X on channels 0-7)"));
  }
}

bool readTofOnChannel(uint8_t idx) {
  if (!tofMuxReady || idx >= TOF_SENSOR_COUNT) {
    tofDistanceMm[idx] = -1;
    tofValid[idx] = false;
    return false;
  }
  if (!tofSensorOk[idx] && !recoverVl53OnMux(idx)) {
    tofDistanceMm[idx] = -1;
    tofValid[idx] = false;
    return false;
  }
  if (!tcaSelect(TOF_MUX_CH[idx])) {
    tofDistanceMm[idx] = -1;
    tofValid[idx] = false;
    return false;
  }
  delay(TOF_MUX_SETTLE_MS);
  uint16_t raw = vl53[idx].readRange();
  int16_t mm = sanitizeTofRangeMm(raw);
  if (mm < 0) {
    tofDistanceMm[idx] = -1;
    tofValid[idx] = false;
    return false;
  }
  if (mm > 0 && mm < (int16_t)TOF_MIN_VALID_MM) {
    tofDistanceMm[idx] = -1;
    tofValid[idx] = false;
    return false;
  }
  tofDistanceMm[idx] = mm;
  tofValid[idx] = (mm > 0);
  return tofValid[idx];
}

void readAllTof() {
  if (!tofMuxReady) {
    for (uint8_t i = 0; i < TOF_SENSOR_COUNT; i++) {
      tofDistanceMm[i] = -1;
      tofValid[i] = false;
    }
    return;
  }
  for (uint8_t i = 0; i < TOF_SENSOR_COUNT; i++) {
    readTofOnChannel(i);
  }
  lastTofSweepMs = millis();
}

void initTofSensors() {
  tofMuxReady = false;
  for (uint8_t i = 0; i < TOF_SENSOR_COUNT; i++) {
    tofSensorOk[i] = false;
    tofDistanceMm[i] = -1;
    tofValid[i] = false;
  }

  if (!findTca9548Addr()) {
    Serial.println(F("WARN TCA9548A not found (0x70-0x77)"));
    printI2cScan();
    return;
  }

  tofMuxReady = true;

  uint8_t okCount = 0;
  for (uint8_t i = 0; i < TOF_SENSOR_COUNT; i++) {
    if (!tcaSelect(TOF_MUX_CH[i])) {
      Serial.print(F("WARN TOF mux ch"));
      Serial.print(TOF_MUX_CH[i]);
      Serial.println(F(" select fail"));
      continue;
    }
    if (!initVl53OnBus(vl53[i])) {
      Serial.print(F("WARN TOF ch"));
      Serial.print(TOF_MUX_CH[i]);
      Serial.println(F(" init fail"));
      continue;
    }
    tofSensorOk[i] = true;
    okCount++;
    Serial.print(F("TOF ch"));
    Serial.print(TOF_MUX_CH[i]);
    Serial.println(F(" init ok"));
  }

  if (okCount == TOF_SENSOR_COUNT) {
    Serial.println(F("TOF READY ch=0,1,2"));
  } else if (okCount > 0) {
    Serial.print(F("TOF READY partial ("));
    Serial.print(okCount);
    Serial.println(F("/3)"));
  } else {
    Serial.println(F("WARN TOF none ready"));
    tofMuxReady = false;
  }
}

bool handleTofCommand() {
  if (lineLen == 0) return false;
  lineBuffer[lineLen] = '\0';

  if (lineLen == 1 && lineBuffer[0] == 'F') {
    readAllTof();
    printTofLine();
    return true;
  }

  if (lineBuffer[0] == 'O') {
    if (lineLen >= 2 && lineBuffer[1] == '0') {
      tofStreamEnabled = false;
      return true;
    }
    if (lineLen >= 2 && lineBuffer[1] == '1') {
      tofStreamEnabled = true;
      if (tofStreamHz < 0.5f) tofStreamHz = 5.0f;
      lastTofStreamMs = 0;
      return true;
    }
    float hz = atof(&lineBuffer[1]);
    if (hz < 0.5f) hz = 0.5f;
    if (hz > 20.0f) hz = 20.0f;
    tofStreamHz = hz;
    tofStreamEnabled = true;
    lastTofStreamMs = 0;
    return true;
  }
  return false;
}

void updateTofStream() {
  if (!tofStreamEnabled || !tofMuxReady) return;
  unsigned long intervalMs = (unsigned long)(1000.0f / tofStreamHz);
  if (intervalMs < 50) intervalMs = 50;
  unsigned long now = millis();
  if (now - lastTofStreamMs < intervalMs) return;
  lastTofStreamMs = now;
  readAllTof();
  printTofLine();
}

void handleLine() {
  if (lineLen == 0) return;
  lineBuffer[lineLen] = '\0';

  if (handleTofCommand()) {
    lineLen = 0;
    return;
  }

  if (lineLen == 1) {
    if (lineBuffer[0] == 'S') {
      runPanSweep();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'Z') {
      zeroBaseReference();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == '?') {
      printStatus();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'H') {
      Serial.println(F("READY"));
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'V') {
      printServoStopPose();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'I') {
      printEncoderPins();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'L') {
      startBaseSpinLeft();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'R') {
      startBaseSpinRight();
      lineLen = 0;
      return;
    }
    if (lineBuffer[0] == 'X') {
      stopBaseMotion();
      lineLen = 0;
      return;
    }
  }

  ParsedCommand cmd;
  parseCommandLine(lineBuffer, cmd);
  if (cmd.hasPan || cmd.hasTilt || cmd.hasBase || cmd.hasRaw || cmd.hasCpd) {
    applyParsedCommand(cmd);
  }
  lineLen = 0;
}

void updateBaseMotor() {
  if (!moveActive) return;

  unsigned long now = millis();

  if (now - moveStartMs > MOVE_TIMEOUT_MS) {
    stopBaseMotion();
    Serial.println(F("ERR B timeout"));
    return;
  }

  long pos = readEncoderCount();
  long error = moveTargetCount - pos;
  long errAbs = labs(error);
  if (errAbs <= ENCODER_TOLERANCE) {
    stopBaseMotion();
    pendingBaseAck = true;
    return;
  }

  if (errAbs < lastErrorAbs - 1
      || labs(pos - lastProgressCount) >= STALL_MIN_PROGRESS) {
    lastProgressCount = pos;
    lastProgressMs = now;
    lastErrorAbs = errAbs;
  } else if (now - lastProgressMs > STALL_MS) {
    stopBaseMotion();
    Serial.println(F("ERR B stall"));
    return;
  }

  // TB6612 wiring: negative PWM increases POS, positive PWM decreases POS.
  if (errAbs > COARSE_ERR_COUNTS) {
    motorDrive((error > 0 ? -COARSE_PWM : COARSE_PWM));
    return;
  }

  int out = (int)(FINE_KP * 0.55f * (float)(-error));
  if (out == 0) {
    out = (error > 0 ? -FINE_MIN_PWM : FINE_MIN_PWM);
  } else if (labs(out) < FINE_MIN_PWM) {
    out = (out > 0 ? FINE_MIN_PWM : -FINE_MIN_PWM);
  }
  motorDrive(out);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(ENC_A_PIN, INPUT);
  pinMode(ENC_B_PIN, INPUT);
  pinMode(MOTOR_AIN1_PIN, OUTPUT);
  pinMode(MOTOR_AIN2_PIN, OUTPUT);

  syncEncoderState();
  // Encoder polled in loop() only — CHANGE ISRs on GPIO 34/35 without pull-ups
  // can interrupt-storm and watchdog-reset the chip.

  ledcSetup(LEDC_PWM_CHANNEL, LEDC_FREQ_HZ, LEDC_RES_BITS);
  ledcAttachPin(MOTOR_PWM_PIN, LEDC_PWM_CHANNEL);
  motorStop();

  Serial.begin(115200);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);
  Wire.setTimeOut(50);
  Wire.beginTransmission(0x40);
  if (Wire.endTransmission() == 0) {
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50);
    delay(10);
  } else {
    Serial.println(F("WARN PCA9685 not found at 0x40"));
  }

  for (uint8_t i = 0; i < TOF_SENSOR_COUNT; i++) {
    tofSensorOk[i] = false;
    tofDistanceMm[i] = -1;
    tofValid[i] = false;
  }
  tofMuxReady = findTca9548Addr();
  if (!tofMuxReady) {
    Serial.println(F("WARN TCA9548A not found (0x70-0x77)"));
  }

  // Servo pulses deferred until first P/T (reduces USB brownout on boot).
  panAngle = PAN_CENTER;
  tiltAngle = TILT_CENTER;
  zeroOffset = 0;
  lastCommandMs = millis();
  Serial.println(F("FW head_servo+tof"));
  Serial.println(F("READY"));
  Serial.flush();
  // VL53 sensors init lazily on first F poll (begin() can block a long time on I2C).
}

void loop() {
  serviceEncoderLocked();

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handleLine();
    } else if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuffer[lineLen++] = c;
    }
  }

  updateBaseMotor();

  if (spinPwm != 0) {
    motorDrive(spinPwm);
  }

  if (pendingBaseAck) {
    pendingBaseAck = false;
    printBaseAck(ackBaseDeg);
  }

  updateTofStream();

  delay(1);  // feed task watchdog (tight loop otherwise resets chip)
}

