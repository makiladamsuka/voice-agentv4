from livekit.agents import RunContext, function_tool
import socket
import json
import logging

class EmotionTools:
    @function_tool(description="Change the physical robot's eye emotion. Use this to express how you feel when speaking.")
    async def set_emotion(self, emotion_name: str, ctx: RunContext):
        """
        Set the physical emotion state of the robot. 
        Valid emotions include: 'happy', 'sad', 'angry', 'surprised', 'suspicious', 'sleepy', 'bored', 'thinking', 'curious_intense', 'attentive', 'amused'.
        """
        logging.info(f"Setting robot emotion to: {emotion_name}")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            payload = json.dumps({
                "command": "emotion",
                "emotion": emotion_name
            }).encode("utf-8")
            sock.sendto(payload, ("127.0.0.1", 9000))
        except Exception as e:
            logging.error(f"Failed to send UDP command: {e}")
            
        return f"Successfully set physical emotion to {emotion_name}"
