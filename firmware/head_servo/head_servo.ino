/*
 * Robot Nano: head pan/tilt servos + encoder base motor over USB serial.
 *
 * Head: D9 pan, D10 tilt (external 5V servo power).
 * Base: D2/D3 encoder, D5 PWM, D6/D7 direction -> TB6612FNG -> N20 motor.
 * Gear 11:110 (10:1). CPD set at runtime via C command after manual calibration.
 *
 * Protocol (newline-terminated):
 *   P85.0 T105.0       head only
 *   B+45.0 / B-30.0    base relative degrees
 *   B45.0              base absolute degrees (zero at boot or Z)
 *   P85.0 T105.0 B+25  combined
 *   M1000 / M-1000     raw units; value/10 = encoder counts delta
 *   C1.222             set counts per degree (runtime until reboot)
 *   Z                  zero encoder + angle reference
 *   ?                  status: POS <n> DEG <f> CPD <f> BUSY 0|1
 *   I                  raw encoder pins: ENC A=0|1 B=0|1 POS <n>
 *   S                  onboard pan servo sweep (bench)
 */

#include <Servo.h>
#include <string.h>

#define DEBUG_ACK 1

// --- Head servos ---
const int PAN_PIN = 9;
const int TILT_PIN = 10;
const float PAN_MIN = 40.0f;
const float PAN_MAX = 130.0f;
const float TILT_MIN = 80.0f;
const float TILT_MAX = 130.0f;
const float PAN_CENTER = 85.0f;
const float TILT_CENTER = 105.0f;
const int PULSE_MIN_US = 450;
const int PULSE_MAX_US = 2600;

// --- Base motor / encoder ---
const int ENC_A_PIN = 2;
const int ENC_B_PIN = 3;
const int MOTOR_PWM_PIN = 5;
const int MOTOR_AIN1_PIN = 6;
const int MOTOR_AIN2_PIN = 7;

// Boot placeholder — real CPD comes from manual calibration (C command).
float countsPerBaseDeg = 1.0f;
const int ENCODER_TOLERANCE = 8;
const int PWM_MAX = 200;
const unsigned long PID_INTERVAL_MS = 15;
const unsigned long MOVE_TIMEOUT_MS = 15000;

const float PID_KP = 2.8f;
const float PID_KI = 0.04f;
const float PID_KD = 0.5f;

const int LED_PIN = LED_BUILTIN;
const int LINE_BUF_SIZE = 64;

struct ParsedCommand {
  bool hasPan;
  bool hasTilt;
  bool hasBase;
  bool hasRaw;
  bool hasCpd;
  bool baseRelative;
  float pan;
  float tilt;
  float baseDeg;
  long rawUnits;
  float cpdValue;
};

Servo panServo;
Servo tiltServo;

float panAngle = PAN_CENTER;
float tiltAngle = TILT_CENTER;
unsigned long lastCommandMs = 0;

char lineBuffer[LINE_BUF_SIZE];
uint8_t lineLen = 0;

volatile long encoderCount = 0;
uint8_t encLastState = 0;

// Gray-code quadrature table: index = (prevState << 2) | newState
const int8_t ENC_QUAD_TABLE[16] = {
  0, 1, -1, 0,
  -1, 0, 0, 1,
  1, 0, 0, -1,
  0, -1, 1, 0
};

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

void syncEncoderState() {
  uint8_t a = digitalRead(ENC_A_PIN) & 1;
  uint8_t b = digitalRead(ENC_B_PIN) & 1;
  encLastState = (a << 1) | b;
}
long zeroOffset = 0;
long moveTargetCount = 0;
bool baseBusy = false;
bool moveActive = false;
unsigned long moveStartMs = 0;
unsigned long lastPidMs = 0;
float pidIntegral = 0.0f;
long lastEncoderForPid = 0;
float ackBaseDeg = 0.0f;
bool pendingBaseAck = false;

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

int mapAngleToUs(float deg, float degMin, float degMax) {
  float span = degMax - degMin;
  if (span < 0.001f) return PULSE_MIN_US;
  float t = (deg - degMin) / span;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return PULSE_MIN_US + (int)(t * (float)(PULSE_MAX_US - PULSE_MIN_US));
}

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

void encAISR() {
  serviceEncoder();
}

void encBISR() {
  serviceEncoder();
}

void motorStop() {
  analogWrite(MOTOR_PWM_PIN, 0);
  digitalWrite(MOTOR_AIN1_PIN, LOW);
  digitalWrite(MOTOR_AIN2_PIN, LOW);
}

void motorDrive(int pwm) {
  pwm = constrain(pwm, -PWM_MAX, PWM_MAX);
  if (pwm == 0) {
    motorStop();
    return;
  }
  if (pwm > 0) {
    digitalWrite(MOTOR_AIN1_PIN, HIGH);
    digitalWrite(MOTOR_AIN2_PIN, LOW);
    analogWrite(MOTOR_PWM_PIN, pwm);
  } else {
    digitalWrite(MOTOR_AIN1_PIN, LOW);
    digitalWrite(MOTOR_AIN2_PIN, HIGH);
    analogWrite(MOTOR_PWM_PIN, -pwm);
  }
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
  panServo.writeMicroseconds(mapAngleToUs(pan, PAN_MIN, PAN_MAX));
  tiltServo.writeMicroseconds(mapAngleToUs(tilt, TILT_MIN, TILT_MAX));
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(LED_PIN, LOW);
  if (emitAck) printServoAck(pan, tilt);
}

bool startBaseMoveToCount(long targetCount, float ackDeg) {
  if (baseBusy) {
    printBaseBusy();
    return false;
  }
  moveTargetCount = targetCount;
  ackBaseDeg = ackDeg;
  pidIntegral = 0.0f;
  lastEncoderForPid = encoderCount;
  moveActive = true;
  baseBusy = true;
  moveStartMs = millis();
  return true;
}

bool startBaseRelativeDeg(float deltaDeg) {
  long deltaCounts = (long)(deltaDeg * countsPerBaseDeg);
  long target = encoderCount + deltaCounts;
  float targetDeg = countsToDeg(target);
  return startBaseMoveToCount(target, targetDeg);
}

bool startBaseAbsoluteDeg(float deg) {
  long target = degToCounts(deg);
  return startBaseMoveToCount(target, deg);
}

bool startBaseRawUnits(long units) {
  long deltaCounts = units / 10;
  long target = encoderCount + deltaCounts;
  float targetDeg = countsToDeg(target);
  return startBaseMoveToCount(target, targetDeg);
}

void zeroBaseReference() {
  if (baseBusy) {
    printBaseBusy();
    return;
  }
  noInterrupts();
  encoderCount = 0;
  zeroOffset = 0;
  syncEncoderState();
  interrupts();
  moveTargetCount = 0;
  ackBaseDeg = 0.0f;
  Serial.println(F("OK Z"));
}

void printEncoderPins() {
  serviceEncoder();
  uint8_t a = digitalRead(ENC_A_PIN) & 1;
  uint8_t b = digitalRead(ENC_B_PIN) & 1;
  noInterrupts();
  long pos = encoderCount;
  interrupts();
  Serial.print(F("ENC A="));
  Serial.print(a);
  Serial.print(F(" B="));
  Serial.print(b);
  Serial.print(F(" POS "));
  Serial.println(pos);
}

void printStatus() {
  noInterrupts();
  long pos = encoderCount;
  interrupts();
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
  cmd.baseRelative = false;
  cmd.pan = PAN_CENTER;
  cmd.tilt = TILT_CENTER;
  cmd.baseDeg = 0.0f;
  cmd.rawUnits = 0;
  cmd.cpdValue = 0.0f;

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

void handleLine() {
  if (lineLen == 0) return;
  lineBuffer[lineLen] = '\0';

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
    if (lineBuffer[0] == 'I') {
      printEncoderPins();
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
  if (now - lastPidMs < PID_INTERVAL_MS) return;
  lastPidMs = now;

  if (now - moveStartMs > MOVE_TIMEOUT_MS) {
    motorStop();
    moveActive = false;
    baseBusy = false;
    Serial.println(F("ERR B timeout"));
    return;
  }

  noInterrupts();
  long pos = encoderCount;
  interrupts();

  long error = moveTargetCount - pos;
  if (labs(error) <= ENCODER_TOLERANCE) {
    motorStop();
    moveActive = false;
    baseBusy = false;
    pendingBaseAck = true;
    return;
  }

  float dt = PID_INTERVAL_MS / 1000.0f;
  pidIntegral += (float)error * dt;
  pidIntegral = constrain(pidIntegral, -400.0f, 400.0f);
  long dError = pos - lastEncoderForPid;
  lastEncoderForPid = pos;
  float derivative = -(float)dError / dt;

  float out = PID_KP * (float)error + PID_KI * pidIntegral + PID_KD * derivative;
  motorDrive((int)out);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);
  pinMode(MOTOR_AIN1_PIN, OUTPUT);
  pinMode(MOTOR_AIN2_PIN, OUTPUT);
  pinMode(MOTOR_PWM_PIN, OUTPUT);

  syncEncoderState();
  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encAISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B_PIN), encBISR, CHANGE);
  motorStop();

  Serial.begin(115200);
  panServo.attach(PAN_PIN, PULSE_MIN_US, PULSE_MAX_US);
  tiltServo.attach(TILT_PIN, PULSE_MIN_US, PULSE_MAX_US);
  writeAngles(PAN_CENTER, TILT_CENTER, false);
  zeroOffset = 0;
  lastCommandMs = millis();
  Serial.println(F("READY"));
}

void loop() {
  serviceEncoder();

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handleLine();
    } else if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuffer[lineLen++] = c;
    }
  }

  updateBaseMotor();

  if (pendingBaseAck) {
    pendingBaseAck = false;
    printBaseAck(ackBaseDeg);
  }
}
