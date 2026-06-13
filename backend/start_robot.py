import subprocess
import sys
import time

def start_services():
    print("🚀 Starting Dual Robot Services...")

    print("👀 Starting TFT Eyes & Face Tracker...")
    print("   Debug dashboard: http://<pi-ip>:8080/debug")
    eyes_proc = subprocess.Popen([sys.executable, "robot_eyes.py"])

    print("🗣️ Starting LiveKit Voice Agent (joins when frontend connects)...")
    voice_proc = subprocess.Popen([sys.executable, "voice_agent.py", "dev"])
    
    try:
        # Keep the main thread alive watching both
        while True:
            # Check if either process randomly crashed
            if eyes_proc.poll() is not None:
                print("⚠️  Warning: Eyes process exited unexpectedly.")
                break
                
            if voice_proc.poll() is not None:
                print("⚠️  Warning: Voice Agent process exited unexpectedly.")
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested! Gracefully killing both services...")
    finally:
        # Cleanly terminate child processes
        print("🛑 Sending termination signals...")
        if eyes_proc.poll() is None:
            eyes_proc.terminate()
        if voice_proc.poll() is None:
            voice_proc.terminate()
            
        print("⏳ Waiting for processes to die (eyes homing servos)...")
        try:
            eyes_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            eyes_proc.kill()
        try:
            voice_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            voice_proc.kill()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    start_services()
