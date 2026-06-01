"""USB serial transport for Arduino Nano (head servos + base motor)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import serial
except ImportError:
    serial = None  # type: ignore

DEFAULT_PORTS = ("/dev/ttyUSB0", "/dev/ttyACM0")
READY_TIMEOUT_SEC = 2.0
POST_CONNECT_DELAY_SEC = 1.2
MIN_SEND_INTERVAL_SEC = 0.02
ACK_TIMEOUT_SEC = 1.0
BASE_MOVE_TIMEOUT_SEC = 15.0
BOOT_CPD = 1.0

_SERVO_ACK_RE = re.compile(r"^OK\s+P(\d+)\s+T(\d+)\s*$")
_BASE_ACK_RE = re.compile(r"^OK\s+B(-?\d+(?:\.\d+)?)\s*$")
_OK_C_RE = re.compile(r"^OK\s+C(-?\d+(?:\.\d+)?)\s*$")
_STATUS_RE = re.compile(
    r"^POS\s+(-?\d+)\s+DEG\s+(-?\d+(?:\.\d+)?)\s+CPD\s+(-?\d+(?:\.\d+)?)\s+BUSY\s+([01])\s*$"
)
_ENC_PINS_RE = re.compile(r"^ENC A=([01]) B=([01]) POS (-?\d+)\s*$")


@dataclass
class EncoderPins:
    a: int
    b: int
    encoder_count: int


@dataclass
class BaseStatus:
    encoder_count: int
    degrees: float
    counts_per_degree: float
    busy: bool


def _port_missing(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 2:
        return True
    msg = str(exc).lower()
    return "no such file" in msg or "errno 2" in msg


def resolve_port(port: str) -> list[str]:
    if port:
        return [port]
    return list(DEFAULT_PORTS)


class ArduinoServoLink:
    """Serial link to robot Nano firmware (servos + base)."""

    def __init__(self, port: str = "", baud: int = 115200):
        self._port_name = port
        self._baud = baud
        self._ser: Optional[serial.Serial] = None
        self._connected = False
        self._last_pan: Optional[float] = None
        self._last_tilt: Optional[float] = None
        self._last_send_ts = 0.0
        self._error_logged = False
        self._last_ack: Optional[Tuple[int, int]] = None
        self._last_base_ack: Optional[float] = None
        self.base_move_timeout_sec = BASE_MOVE_TIMEOUT_SEC

    @property
    def connected(self) -> bool:
        return self._connected and self._ser is not None

    def _drain_rx(self) -> None:
        if self._ser is not None and self._ser.in_waiting:
            self._ser.read(self._ser.in_waiting)

    def connect(self) -> bool:
        if serial is None:
            print("pyserial not installed; pip install pyserial")
            return False
        ports = resolve_port(self._port_name)
        for i, port in enumerate(ports):
            try:
                self._ser = serial.Serial(port, self._baud, timeout=0.05)
                self._ser.reset_input_buffer()
                time.sleep(POST_CONNECT_DELAY_SEC)
                deadline = time.time() + READY_TIMEOUT_SEC
                while time.time() < deadline:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if line == "READY":
                        self._connected = True
                        self._error_logged = False
                        self._last_pan = None
                        self._last_tilt = None
                        self._drain_rx()
                        print(f"Arduino robot nano ready on {port}")
                        return True
                    time.sleep(0.05)
                print(f"Arduino on {port}: no READY within {READY_TIMEOUT_SEC}s")
                self.close()
            except Exception as e:
                self.close()
                if self._port_name:
                    if not self._error_logged:
                        print(f"Arduino serial connect failed ({port}): {e}")
                        self._error_logged = True
                    return False
                if _port_missing(e) and i < len(ports) - 1:
                    continue
                if not self._error_logged:
                    if _port_missing(e) and len(ports) > 1:
                        print(
                            f"Arduino serial connect failed: no device on {', '.join(ports)}. "
                            "Plug in the Nano or set arduino_port (e.g. /dev/ttyUSB0)."
                        )
                    else:
                        print(f"Arduino serial connect failed ({port}): {e}")
                    self._error_logged = True
        return False

    def _read_lines_until(self, timeout: float, matchers: tuple) -> Optional[str]:
        if not self._connected or self._ser is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            for pattern in matchers:
                if pattern.match(line):
                    return line
            if line.startswith("ERR B"):
                return line
        return None

    def read_ack(self, timeout: float = ACK_TIMEOUT_SEC) -> Optional[Tuple[int, int]]:
        line = self._read_lines_until(timeout, (_SERVO_ACK_RE,))
        if line is None:
            return None
        match = _SERVO_ACK_RE.match(line)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def read_base_ack(self, timeout: Optional[float] = None) -> Optional[float]:
        if timeout is None:
            timeout = self.base_move_timeout_sec
        line = self._read_lines_until(timeout, (_BASE_ACK_RE,))
        if line is None:
            return None
        if line.startswith("ERR B"):
            print(line)
            return None
        match = _BASE_ACK_RE.match(line)
        if match:
            return float(match.group(1))
        return None

    def send_line(
        self,
        payload: str,
        *,
        wait_base: bool = False,
        wait_servo: bool = False,
        drain_after: bool = True,
    ) -> bool:
        if not self._connected or self._ser is None:
            return False
        try:
            self._drain_rx()
            self._ser.write(payload.encode("ascii"))
            if not payload.endswith("\n"):
                self._ser.write(b"\n")
            self._ser.flush()
            if wait_servo:
                self._last_ack = self.read_ack()
            if wait_base:
                self._last_base_ack = self.read_base_ack()
            elif drain_after and not wait_servo:
                self._drain_rx()
            return True
        except Exception as e:
            if not self._error_logged:
                print(f"Arduino serial write failed: {e}")
                self._error_logged = True
            self._connected = False
            return False

    def send_raw(self, payload: bytes, wait_ack: bool = False) -> Optional[Tuple[int, int]]:
        if not self._connected or self._ser is None:
            return None
        try:
            self._drain_rx()
            self._ser.write(payload)
            self._ser.flush()
            if wait_ack:
                return self.read_ack()
            self._drain_rx()
            return None
        except Exception as e:
            if not self._error_logged:
                print(f"Arduino serial write failed: {e}")
                self._error_logged = True
            self._connected = False
            return None

    def run_bench_sweep(self) -> bool:
        if not self.send_raw(b"S\n", wait_ack=False):
            return False
        time.sleep(6.5)
        self._drain_rx()
        self._last_pan = None
        self._last_tilt = None
        return True

    def write_angles(
        self,
        pan: float,
        tilt: float,
        *,
        force: bool = False,
        wait_ack: bool = False,
    ) -> bool:
        if not self._connected or self._ser is None:
            return False
        now = time.time()
        changed = (
            self._last_pan is None
            or self._last_tilt is None
            or abs(pan - self._last_pan) > 0.05
            or abs(tilt - self._last_tilt) > 0.05
        )
        if not force and not changed and (now - self._last_send_ts) < MIN_SEND_INTERVAL_SEC:
            return True
        ok = self.send_line(f"P{pan:.1f} T{tilt:.1f}", wait_servo=wait_ack)
        if ok:
            self._last_pan = pan
            self._last_tilt = tilt
            self._last_send_ts = now
        return ok

    def set_counts_per_degree(self, cpd: float) -> bool:
        ok = self.send_line(f"C{cpd:.4f} P85.0 T105.0", drain_after=False)
        if ok:
            self._read_lines_until(ACK_TIMEOUT_SEC, (_OK_C_RE,))
            self._read_lines_until(0.3, (_SERVO_ACK_RE,))
        st = self.query_status() if ok else None
        return st is not None and abs(st.counts_per_degree - cpd) < 0.05

    def is_calibrated(self) -> bool:
        st = self.query_status()
        return st is not None and abs(st.counts_per_degree - BOOT_CPD) > 0.05

    def write_base_relative(self, deg: float, *, wait: bool = True) -> bool:
        sign = "+" if deg >= 0 else ""
        return self.send_line(f"B{sign}{deg:.1f}", wait_base=wait)

    def write_base_absolute(self, deg: float, *, wait: bool = True) -> bool:
        return self.send_line(f"B{deg:.1f}", wait_base=wait)

    def write_base_raw(self, units: int, *, wait: bool = True) -> bool:
        sign = "+" if units >= 0 else ""
        return self.send_line(f"M{sign}{units}", wait_base=wait)

    def write_combined(
        self,
        pan: float,
        tilt: float,
        base_rel: Optional[float] = None,
        *,
        wait_servo: bool = False,
        wait_base: bool = True,
    ) -> bool:
        parts = [f"P{pan:.1f}", f"T{tilt:.1f}"]
        if base_rel is not None:
            sign = "+" if base_rel >= 0 else ""
            parts.append(f"B{sign}{base_rel:.1f}")
        line = " ".join(parts)
        ok = self.send_line(
            line,
            wait_servo=wait_servo or base_rel is not None,
            wait_base=wait_base and base_rel is not None,
        )
        if ok:
            self._last_pan = pan
            self._last_tilt = tilt
            self._last_send_ts = time.time()
        return ok

    def zero_base(self) -> bool:
        return self.send_line("Z")

    def query_status(self) -> Optional[BaseStatus]:
        if not self.send_line("?", drain_after=False):
            return None
        deadline = time.time() + ACK_TIMEOUT_SEC
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()  # type: ignore
            if not line:
                continue
            match = _STATUS_RE.match(line)
            if match:
                return BaseStatus(
                    encoder_count=int(match.group(1)),
                    degrees=float(match.group(2)),
                    counts_per_degree=float(match.group(3)),
                    busy=match.group(4) == "1",
                )
        return None

    def query_encoder_pins(self) -> Optional[EncoderPins]:
        if not self.send_line("I", drain_after=False):
            return None
        deadline = time.time() + ACK_TIMEOUT_SEC
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()  # type: ignore
            if not line:
                continue
            match = _ENC_PINS_RE.match(line)
            if match:
                return EncoderPins(
                    a=int(match.group(1)),
                    b=int(match.group(2)),
                    encoder_count=int(match.group(3)),
                )
        return None

    def close(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.write(b"P85.0 T105.0\n")
                    time.sleep(0.05)
                    self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._connected = False
