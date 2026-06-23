import os
import signal
import subprocess
import sys
import time

# Captured arm home (raise low + min sweep) — matches voice-agentv5 calibration.
ARM_HOME_DEG: dict[str, float] = {
    "arm_0": 47.0,
    "arm_1": 65.0,
    "arm_2": 64.0,
    "arm_3": 87.0,
}


def _apply_arm_home_env() -> None:
    """Pass arm home pose to robot_eyes via environment before subprocess start."""
    for arm, deg in ARM_HOME_DEG.items():
        os.environ[f"ROBOT_{arm.upper()}_HOME"] = f"{deg:.1f}"

def _kill_stale_robot_processes() -> None:
    """Free ports 8080/9001 left by an unclean Ctrl+C or crashed start_robot."""
    me = os.getpid()
    patterns = ("voice_agent.py", "robot_eyes.py")
    killed: list[int] = []
    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        for token in result.stdout.split():
            if not token.strip().isdigit():
                continue
            pid = int(token)
            if pid == me:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                pass
    if not killed:
        return
    print(f"Stopped stale robot process(es): {', '.join(str(p) for p in killed)}")
    time.sleep(1.0)
    for pid in killed:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _terminate_process_group(proc: subprocess.Popen, *, label: str, grace_sec: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        proc.wait(timeout=2)


def start_services():
    _kill_stale_robot_processes()
    _apply_arm_home_env()

    print("Starting Dual Robot Services...")
    print("   Robot debug:     http://<pi-ip>:8090/debug")
    print("   Frontend assets: http://<pi-ip>:8080/assets/ (voice agent, background)")
    print(
        f"   Arm home:        A0={ARM_HOME_DEG['arm_0']:.0f} "
        f"A1={ARM_HOME_DEG['arm_1']:.0f} "
        f"A2={ARM_HOME_DEG['arm_2']:.0f} "
        f"A3={ARM_HOME_DEG['arm_3']:.0f}"
    )

    print("Starting TFT Eyes & Face Tracker (priority)...")
    eyes_proc = subprocess.Popen(
        [sys.executable, "robot_eyes.py"],
        start_new_session=True,
    )

    print("Starting LiveKit Voice Agent in background (DB index, media server)...")
    voice_proc = subprocess.Popen(
        [sys.executable, "voice_agent.py", "dev"],
        start_new_session=True,
    )

    try:
        while True:
            if eyes_proc.poll() is not None:
                print("Warning: Eyes process exited unexpectedly.")
                break

            if voice_proc.poll() is not None:
                print("Warning: Voice Agent process exited unexpectedly.")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutdown requested! Gracefully killing both services...")
    finally:
        print("Sending termination signals...")
        _terminate_process_group(eyes_proc, label="eyes", grace_sec=5)
        _terminate_process_group(voice_proc, label="voice", grace_sec=3)
        print("Shutdown complete.")


if __name__ == "__main__":
    start_services()
