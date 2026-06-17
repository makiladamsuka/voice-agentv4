"""Shared ESP32 USB-serial policy for robot runtime and test scripts.

Do not change these defaults without updating .cursor/rules/esp32-serial.mdc
and verifying test_head_servos.py + start_robot.py together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from arduino_servo import ArduinoServoLink

# ESP32 cannot process ~100 P/T lines/sec; coalesce sends to this rate.
ESP32_SERIAL_SEND_HZ = 30.0

# Wait for /dev/ttyUSB0 hotplug before giving up.
ESP32_PORT_WAIT_SEC = 45.0

# Skip slow ToF probe at connect; tof_worker polls after link is live.
ESP32_SKIP_TOF_PROBE_AT_CONNECT = True

# Never DTR-reset ESP32 during robot/test reconnect (avoids reboot + command backlog).
ESP32_ALLOW_DTR_RESET = False


def configure_esp32_link(link: ArduinoServoLink) -> None:
    """Apply the standard connect policy before link.connect()."""
    link._skip_tof_probe_at_connect = ESP32_SKIP_TOF_PROBE_AT_CONNECT
    link._allow_dtr_reset = ESP32_ALLOW_DTR_RESET
    link._port_wait_sec = ESP32_PORT_WAIT_SEC


def prepare_esp32_for_live_control(link: ArduinoServoLink) -> None:
    """Drain queued commands and stop base before head tracking / interactive control."""
    link.flush_pending_commands()
    link.write_base_stop()
    link.set_tof_stream(False)


def connect_esp32(
    port: str = "",
    baud: int = 115200,
    *,
    prepare: bool = True,
) -> Optional[ArduinoServoLink]:
    """Open ESP32 with standard policy; returns None on failure."""
    from arduino_servo import ArduinoServoLink

    link = ArduinoServoLink(port=port, baud=baud)
    configure_esp32_link(link)
    if not link.connect():
        return None
    if prepare:
        prepare_esp32_for_live_control(link)
    return link


def serial_send_interval_sec() -> float:
    return 1.0 / max(5.0, ESP32_SERIAL_SEND_HZ)
