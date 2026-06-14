import subprocess
import sys
import time
from pathlib import Path

EYES_START_DELAY_SEC = 15


def _warm_event_database():
    """Index posters on disk before the LiveKit worker starts."""
    print("Indexing event posters (runs once at robot startup)...")
    try:
        from event_database import build_event_database

        assets_dir = Path(__file__).parent / "assets"
        build_event_database(assets_dir)
        print("Event database ready")
    except Exception as exc:
        print(f"Event indexing failed (worker will retry in prewarm): {exc}")


def start_services():
    print("Starting Dual Robot Services...")
    print("   Frontend assets: http://<pi-ip>:8080/assets/")
    print("   Robot debug:     http://<pi-ip>:8090/debug")

    _warm_event_database()

    print("Starting LiveKit Voice Agent (prewarming worker, image server on :8080)...")
    voice_proc = subprocess.Popen([sys.executable, "voice_agent.py", "dev"])

    print(
        f"Waiting {EYES_START_DELAY_SEC}s for frontend to connect before loading camera/eyes..."
    )
    time.sleep(EYES_START_DELAY_SEC)

    print("Starting TFT Eyes & Face Tracker...")
    eyes_proc = subprocess.Popen([sys.executable, "robot_eyes.py"])

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
        if eyes_proc.poll() is None:
            eyes_proc.terminate()
        if voice_proc.poll() is None:
            voice_proc.terminate()

        print("Waiting for processes to die (eyes homing servos)...")
        try:
            eyes_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            eyes_proc.kill()
        try:
            voice_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            voice_proc.kill()
        print("Shutdown complete.")


if __name__ == "__main__":
    start_services()
