"""USB serial transport for ESP32 robot firmware (PCA9685 head + base motor)."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from tof_presence import TofSnapshot, parse_tof_line

try:
    import serial
except ImportError:
    serial = None  # type: ignore

DEFAULT_PORTS = ("/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0")
READY_TIMEOUT_SEC = 5.0
POST_CONNECT_DELAY_SEC = 0.5
MIN_SEND_INTERVAL_SEC = 0.01
ACK_TIMEOUT_SEC = 1.0
BASE_MOVE_TIMEOUT_SEC = 15.0
BOOT_CPD = 1.0
BOOT_READ_TIMEOUT_SEC = 12.0
HANDSHAKE_TIMEOUT_SEC = 3.5
STALLED_BOOT_RECOVERY_SEC = 18.0

_SERVO_ACK_RE = re.compile(r"^OK\s+P(\d+)\s+T(\d+)\s*$")
_BASE_ACK_RE = re.compile(r"^OK\s+B(-?\d+(?:\.\d+)?)\s*$")
_OK_C_RE = re.compile(r"^OK\s+C(-?\d+(?:\.\d+)?)\s*$")
_STATUS_RE = re.compile(
    r"^POS\s+(-?\d+)\s+DEG\s+(-?\d+(?:\.\d+)?)\s+CPD\s+(-?\d+(?:\.\d+)?)\s+BUSY\s+([01])\s*$"
)
_TOF_RE = re.compile(r"^TOF\s+")
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


def _is_plausible_serial_line(line: str) -> bool:
    """Drop ESP32 ROM/bootloader fragments that readline splits into noise."""
    if not line or len(line) > 160:
        return False
    lowered = line.lower()
    if lowered.startswith("entry 0x") or lowered.startswith("try 0x"):
        return False
    if lowered.startswith("load:0x") or lowered.startswith("configsip:"):
        return False
    if lowered.startswith("clk_drv:") or lowered.startswith("mode:"):
        return False
    if lowered.startswith("ets ") and "POWERON_RESET" not in line:
        return False
    if "rst:0x" in lowered and "boot:0x" in lowered:
        return False
    printable = sum(1 for c in line if c.isprintable() or c in " \t")
    return printable >= max(4, int(len(line) * 0.75))


def resolve_port(port: str) -> list[str]:
    if port:
        if os.path.exists(port):
            return [port]
        fallbacks = [p for p in DEFAULT_PORTS if os.path.exists(p)]
        print(fallbacks)
        if fallbacks:
            print(
                f"Configured serial port {port} not found; using {fallbacks[0]} instead"
            )
            return fallbacks
        return [port]
    return [p for p in DEFAULT_PORTS if os.path.exists(p)]


class ArduinoServoLink:
    """Serial link to ESP32 firmware (PCA9685 servos + base)."""

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
        self._last_tof: TofSnapshot = TofSnapshot.empty()
        self._tof_capable = False
        self._boot_lines: list[str] = []
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected and self._ser is not None

    def _drain_rx(self) -> None:
        if self._ser is not None and self._ser.in_waiting:
            self._ser.read(self._ser.in_waiting)

    def _esp32_reset(self) -> None:
        """Toggle USB-serial DTR so ESP32 reboots and we capture full boot log."""
        if self._ser is None:
            return
        try:
            self._ser.dtr = False
            self._ser.rts = False
            time.sleep(0.05)
            self._ser.dtr = True
            time.sleep(0.05)
            self._ser.dtr = False
            time.sleep(0.15)
        except Exception:
            pass

    def _note_boot_line(self, line: str) -> None:
        if not _is_plausible_serial_line(line):
            return
        self._boot_lines.append(line)
        if (
            "TOF READY" in line
            or line.startswith("TOF L=")
            or "FW head_servo" in line
            or "WARN TCA9548A" in line
            or "TOF mux @" in line
            or line.startswith("I2C scan:")
            or line.startswith("dev 0x")
        ):
            self._tof_capable = True

    def _wait_for_ready(self, deadline: float) -> bool:
        if self._ser is None:
            return False
        while time.time() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                self._note_boot_line(line)
            if line == "READY":
                return True
            if _STATUS_RE.match(line):
                return True
            time.sleep(0.02)
        return False

    def connect(self) -> bool:
        with self._lock:
            return self._connect_unlocked()

    def _send_command_and_wait(self, cmd: bytes, timeout_sec: float) -> bool:
        if self._ser is None:
            return False
        self._ser.write(cmd)
        self._ser.flush()
        return self._wait_for_ready(time.time() + timeout_sec)

    def _finalize_connect(self, port: str) -> bool:
        self._connected = True
        self._error_logged = False
        self._last_pan = None
        self._last_tilt = None
        if not self._tof_capable:
            try:
                self._ser.write(b"F\n")
                self._ser.flush()
                # First F can block while VL53 sensors init on the ESP32.
                probe = self._read_tof_line(10.0)
            except serial.SerialException:
                probe = None
            if probe is not None:
                self._tof_capable = True
        self._drain_rx()
        print(f"Robot ESP32 ready on {port}")
        if self._boot_lines:
            for ln in self._boot_lines:
                if (
                    ln == "READY"
                    or ln.startswith("FW ")
                    or "TOF" in ln
                    or "WARN" in ln
                    or ln.startswith("I2C scan:")
                    or ln.startswith("dev 0x")
                ):
                    print(f"  {ln}")
        if self._tof_capable:
            mux_ok = any(
                "TOF READY" in ln or "TOF mux @" in ln for ln in self._boot_lines
            )
            if mux_ok:
                print("  ToF firmware: OK (mux + sensors)")
            else:
                print(
                    "  ToF firmware: OK (F works); mux/VL53 not on I2C — see boot log (I2C scan)"
                )
        else:
            print(
                "  ToF firmware: NOT DETECTED — flash latest "
                "firmware/head_servo/head_servo.ino then retry"
            )
        return True

    def _print_no_ready_help(self, port: str) -> None:
        print(f"Arduino on {port}: no READY (timeout {BOOT_READ_TIMEOUT_SEC:.0f}s)")
        if self._boot_lines:
            print("  Serial lines received:")
            for ln in self._boot_lines[-14:]:
                print(f"    {ln}")
        else:
            print("  (no text received — wrong firmware, baud, or USB data line)")
        print(
            "  Fix: flash firmware/head_servo/head_servo.ino, "
            f"then test with: python tests/test_head_servos.py --port {port}"
        )

    def _listen_for_ready(self, deadline: float) -> bool:
        if self._ser is None:
            return False
        while time.time() < deadline:
            if self._wait_for_ready(time.time() + 0.25):
                return True
        return False

    def _recovery_poll_handshake(self, total_sec: float) -> bool:
        """Old firmware can stall in VL53 init after TOF mux line — poll H until loop runs."""
        if self._ser is None:
            return False
        deadline = time.time() + total_sec
        while time.time() < deadline:
            if self._send_command_and_wait(b"H\n", 1.2):
                return True
            time.sleep(0.35)
        return False

    def _stalled_after_mux(self) -> bool:
        if not self._boot_lines:
            return False
        text = "\n".join(self._boot_lines)
        return "TOF mux @" in text and "READY" not in text and "FW head_servo" not in text

    def _attempt_port_connect(self, port: str, *, allow_reset: bool = True) -> bool:
        self._boot_lines = []
        self._tof_capable = False
        self._ser = serial.Serial(
            port,
            self._baud,
            timeout=0.12,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        )
        time.sleep(0.12)

        # 1) Read anything already in the buffer (ESP booted before we opened the port).
        if self._listen_for_ready(time.time() + 1.8):
            return self._finalize_connect(port)

        # 2) Live handshake — do not reset if the chip is already running.
        if self._send_command_and_wait(b"H\n", HANDSHAKE_TIMEOUT_SEC):
            return self._finalize_connect(port)

        if not allow_reset:
            if self._stalled_after_mux():
                if self._recovery_poll_handshake(STALLED_BOOT_RECOVERY_SEC):
                    return self._finalize_connect(port)
            self._print_no_ready_help(port)
            return False

        # 3) One controlled reboot to capture a clean boot log.
        self._ser.reset_input_buffer()
        self._esp32_reset()
        time.sleep(POST_CONNECT_DELAY_SEC + 0.65)
        if self._listen_for_ready(time.time() + BOOT_READ_TIMEOUT_SEC):
            return self._finalize_connect(port)

        # 4) Post-boot commands.
        if self._send_command_and_wait(b"H\n", HANDSHAKE_TIMEOUT_SEC):
            return self._finalize_connect(port)
        if self._send_command_and_wait(b"?\n", HANDSHAKE_TIMEOUT_SEC):
            return self._finalize_connect(port)

        # 5) Boot reached mux scan but never READY (old FW stuck in VL53 init).
        if self._stalled_after_mux():
            print(
                f"Arduino on {port}: boot stalled after ToF mux — "
                f"polling handshake up to {STALLED_BOOT_RECOVERY_SEC:.0f}s "
                "(reflash firmware/head_servo/head_servo.ino for reliable boot)"
            )
            if self._recovery_poll_handshake(STALLED_BOOT_RECOVERY_SEC):
                return self._finalize_connect(port)

        self._print_no_ready_help(port)
        return False

    def _connect_unlocked(self) -> bool:
        if serial is None:
            print("pyserial not installed; pip install pyserial")
            return False

        ports = resolve_port(self._port_name)
        if not ports:
            if self._port_name:
                print(
                    f"Arduino serial connect failed: port {self._port_name!r} not found. "
                    "Plug in the ESP32 USB cable."
                )
            else:
                print(
                    "Arduino serial connect failed: no serial ports found "
                    f"({', '.join(DEFAULT_PORTS)}). Plug in the ESP32."
                )
            return False

        last_err: Optional[BaseException] = None
        for port in ports:
            for attempt in range(3):
                try:
                    # Only the first attempt may DTR-reset; retries must not reboot a booting chip.
                    if self._attempt_port_connect(port, allow_reset=(attempt == 0)):
                        return True
                    self.close()
                    if attempt < 2:
                        time.sleep(1.0)
                except Exception as e:
                    last_err = e
                    self.close()
                    if attempt < 2:
                        time.sleep(0.5)

            if last_err is not None:
                if not self._error_logged:
                    print(f"Arduino serial connect failed ({port}): {last_err}")
                    self._error_logged = True
                if self._port_name:
                    return False

        if last_err is not None and not self._error_logged:
            print(f"Arduino serial connect failed: {last_err}")
            self._error_logged = True
        return False

    def _consume_tof_line(self, line: str) -> bool:
        snap = parse_tof_line(line)
        if snap is not None:
            self._last_tof = snap
            return True
        return False

    def _read_lines_until(self, timeout: float, matchers: tuple) -> Optional[str]:
        if not self._connected or self._ser is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                return None
            if not line:
                continue
            if self._consume_tof_line(line):
                continue
            for pattern in matchers:
                if pattern.match(line):
                    return line
            if line.startswith("ERR B"):
                return line
        return None

    def _read_tof_line(self, timeout: float = 2.0) -> Optional[TofSnapshot]:
        if not self._connected or self._ser is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                return None
            if not line:
                continue
            snap = parse_tof_line(line)
            if snap is not None:
                self._last_tof = snap
                return snap
        return None

    def poll_tof(self, timeout: float = 2.0) -> Optional[TofSnapshot]:
        with self._lock:
            if not self._connected or self._ser is None:
                return None
            try:
                self._ser.write(b"F\n")
                self._ser.flush()
            except Exception as e:
                if not self._error_logged:
                    print(f"ToF poll write failed: {e}")
                    self._error_logged = True
                return None
            return self._read_tof_line(timeout)

    @property
    def tof_capable(self) -> bool:
        return self._tof_capable

    def set_tof_stream(self, enabled: bool, hz: float = 5.0) -> bool:
        if not self._connected or self._ser is None:
            return False
        if enabled:
            hz = max(0.5, min(20.0, hz))
            payload = f"O{hz:.1f}" if abs(hz - round(hz)) > 0.01 else f"O{int(hz)}"
        else:
            payload = "O0"
        return self.send_line(payload, drain_after=True)

    @property
    def last_tof(self) -> TofSnapshot:
        return self._last_tof

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
            self._ensure_base_stopped()
            return None
        match = _BASE_ACK_RE.match(line)
        if match:
            return float(match.group(1))
        return None

    def _ensure_base_stopped(self) -> None:
        """Send stop after firmware ERR B (belt-and-suspenders)."""
        if not self._connected or self._ser is None:
            return
        try:
            self._ser.write(b"X\n")
            self._ser.flush()
        except Exception:
            pass

    def send_line(
        self,
        payload: str,
        *,
        wait_base: bool = False,
        wait_servo: bool = False,
        drain_after: bool = True,
    ) -> bool:
        with self._lock:
            return self._send_line_unlocked(
                payload,
                wait_base=wait_base,
                wait_servo=wait_servo,
                drain_after=drain_after,
            )

    def _send_line_unlocked(
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
                if self._last_base_ack is None:
                    return False
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
        with self._lock:
            return self._send_raw_unlocked(payload, wait_ack=wait_ack)

    def _send_raw_unlocked(
        self, payload: bytes, wait_ack: bool = False
    ) -> Optional[Tuple[int, int]]:
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
            or abs(pan - self._last_pan) > 0.02
            or abs(tilt - self._last_tilt) > 0.02
        )
        if not force and not changed and (now - self._last_send_ts) < MIN_SEND_INTERVAL_SEC:
            return True
        ok = self.send_line(f"P{pan:.1f} T{tilt:.1f}", wait_servo=wait_ack)
        if ok:
            self._last_pan = pan
            self._last_tilt = tilt
            self._last_send_ts = now
        return ok

    def write_servo_frame(
        self,
        values_by_token: dict[str, float],
        *,
        wait_ack: bool = False,
    ) -> bool:
        """
        Send multi-servo command frame, e.g.:
          P85.0 T105.0 A0=90.0 A1=45.0
        """
        if not values_by_token:
            return True
        parts: list[str] = []
        for token, value in values_by_token.items():
            key = token.strip().upper()
            if not key:
                continue
            parts.append(f"{key}{float(value):.1f}")
        if not parts:
            return True
        payload = " ".join(parts)
        ok = self.send_line(payload, wait_servo=wait_ack)
        if ok:
            if "P" in values_by_token:
                self._last_pan = float(values_by_token["P"])
            if "T" in values_by_token:
                self._last_tilt = float(values_by_token["T"])
            self._last_send_ts = time.time()
        return ok

    def set_counts_per_degree(self, cpd: float) -> bool:
        with self._lock:
            ok = self._send_line_unlocked(f"C{cpd:.4f}", drain_after=False)
            if ok:
                self._read_lines_until(ACK_TIMEOUT_SEC, (_OK_C_RE,))
                self._read_lines_until(0.3, (_SERVO_ACK_RE,))
            st = self._query_status_unlocked() if ok else None
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

    def write_base_spin_left(self) -> bool:
        return self.send_line("L", drain_after=False)

    def write_base_spin_right(self) -> bool:
        return self.send_line("R", drain_after=False)

    def write_base_stop(self) -> bool:
        return self.send_line("X", drain_after=False)

    def query_status(self) -> Optional[BaseStatus]:
        with self._lock:
            return self._query_status_unlocked()

    def _query_status_unlocked(self) -> Optional[BaseStatus]:
        if not self._send_line_unlocked("?", drain_after=False):
            return None
        deadline = time.time() + ACK_TIMEOUT_SEC
        while time.time() < deadline:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()  # type: ignore
            except serial.SerialException:
                return None
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
        with self._lock:
            if not self._send_line_unlocked("I", drain_after=False):
                return None
            deadline = time.time() + ACK_TIMEOUT_SEC
            while time.time() < deadline:
                try:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()  # type: ignore
                except serial.SerialException:
                    return None
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

    def write_home_pose(
        self,
        pan: float,
        tilt: float,
        arm_neutrals: Optional[dict[str, float]] = None,
        *,
        wait_ack: bool = True,
    ) -> bool:
        """Send one frame moving head + arms to home/neutral degrees."""
        frame: dict[str, float] = {"P": pan, "T": tilt}
        if arm_neutrals:
            for i in range(4):
                key = f"arm_{i}"
                if key in arm_neutrals:
                    frame[f"A{i}="] = float(arm_neutrals[key])
        return self.write_servo_frame(frame, wait_ack=wait_ack)

    def close(
        self,
        *,
        home_pan: Optional[float] = None,
        home_tilt: Optional[float] = None,
        arm_neutrals: Optional[dict[str, float]] = None,
        skip_home: bool = False,
    ) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    if self._ser.is_open:
                        try:
                            self._send_line_unlocked("O0", drain_after=False)
                        except Exception:
                            pass
                        self._send_line_unlocked("X", drain_after=False)
                        time.sleep(0.02)
                        if not skip_home and home_pan is not None and home_tilt is not None:
                            parts = [f"P{home_pan:.1f}", f"T{home_tilt:.1f}"]
                            if arm_neutrals:
                                for i in range(4):
                                    key = f"arm_{i}"
                                    if key in arm_neutrals:
                                        parts.append(f"A{i}={arm_neutrals[key]:.1f}")
                            self._send_line_unlocked(" ".join(parts), wait_servo=True)
                            time.sleep(0.35)
                        elif not skip_home:
                            self._ser.write(b"P85.0 T105.0\n")
                            time.sleep(0.08)
                        self._ser.close()
                except Exception:
                    pass
            self._ser = None
            self._connected = False
