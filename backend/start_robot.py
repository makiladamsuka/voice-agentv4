import subprocess
import sys
import time

def start_services():
    print("🚀 Starting Dual Robot Services...")
    
    # 1. Start the LiveKit voice agent service FIRST
    print("🗣️ Starting LiveKit Voice Agent and generating Server URL...")
    voice_proc = subprocess.Popen([sys.executable, "voice_agent.py", "dev"])
    
    # Give you 20 seconds to open the link and connect on your laptop
    # before we blast the Pi's CPU with the camera vision model!
    print("⏳ Waiting 20 seconds for you to connect frontend before loading eyes...")
    time.sleep(20)

    # 2. Start the eyes and face tracking service
    print("👀 Starting TFT Eyes & Face Tracker out-of-process...")
    eyes_proc = subprocess.Popen([sys.executable, "robot_eyes.py"])
    
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
            
        print("⏳ Waiting for processes to die...")
        eyes_proc.wait()
        voice_proc.wait()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    start_services()
