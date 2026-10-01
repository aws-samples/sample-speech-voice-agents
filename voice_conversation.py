"""
Voice Conversation Agent - Turn-Based Voice Chat

A complete voice conversation agent that combines text input with
spoken output using Amazon Polly. Streams audio directly to speakers
without saving to disk.

Prerequisites:
    pip install -r requirements.txt

Learning objectives:
- Understand turn-based voice conversation patterns
- See how to combine text input with speech output
- Learn about voice selection and conversation flow
"""

import sys
import subprocess
from shared.model import get_model, get_region
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("Error: boto3 not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


SYSTEM_PROMPT = """You are a friendly conversational assistant.

Guidelines for spoken responses:
- Keep responses to 1-3 sentences for natural speech
- Use conversational, warm language
- Avoid technical jargon unless asked
- Don't use bullet points, code, or special formatting
- Spell out numbers and abbreviations naturally
- End with a question to keep the conversation flowing when appropriate

You're having a casual conversation, so be personable and engaging."""


def _play_audio_stream(audio_bytes: bytes) -> None:
    """Play MP3 audio bytes through the system audio without saving to disk.
    
    Uses ffplay (bundled with ffmpeg) to stream audio from stdin.
    Falls back to afplay/mpg123 with stdin pipe on unsupported systems.
    """
    # Try ffplay first (cross-platform, no file needed)
    try:
        subprocess.run(  # nosec B603 - static argv; audio_bytes is stdin data, not a command
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
            input=audio_bytes,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Fallback: macOS afplay via stdin
    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", "/dev/stdin"], input=audio_bytes, check=True)  # nosec B603 - static argv; stdin data only
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    
    # Fallback: Linux mpg123 via stdin
    try:
        subprocess.run(["mpg123", "-q", "-"], input=audio_bytes, check=True)  # nosec B603 - static argv; stdin data only
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    print("\n[Audio playback failed - install ffmpeg or mpg123 for audio output]")


class VoiceConversationAgent:
    """Turn-based voice conversation using text input and streaming spoken output."""
    
    def __init__(self, voice: str = "Joanna", region: str | None = None):
        """Initialize the voice conversation agent.
        
        Args:
            voice: Polly voice ID
            region: AWS region
        """
        self.region = region or get_region()
        self.polly = boto3.client('polly', region_name=self.region)
        # Streaming handler so the text appears as the model generates it,
        # before Polly speaks the final assembled response.
        self.stream_handler = StreamingCallbackHandler()
        self.agent = Agent(
            model=get_model(),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=self.stream_handler,
        )
        self.voice = voice
        self.speak_enabled = True
    
    def speak(self, text: str) -> bool:
        """Convert text to speech and play it immediately (no disk writes).
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful
        """
        if not self.speak_enabled:
            return False
        
        try:
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=self.voice,
                Engine='neural'
            )
            
            audio_bytes = response['AudioStream'].read()
            _play_audio_stream(audio_bytes)
            return True
            
        except ClientError as e:
            print(f"Speech error: {e}")
            return False
    
    def chat(self, user_input: str) -> str:
        """Stream the response and speak it.
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's text response (assembled from streamed tokens)
        """
        self.stream_handler.reset()
        print("Agent: ", end="", flush=True)
        response = self.agent(user_input)
        print()  # close the streamed line
        response_text = str(response)
        self.speak(response_text)
        return response_text
    
    def toggle_voice(self) -> bool:
        """Toggle voice output on/off."""
        self.speak_enabled = not self.speak_enabled
        return self.speak_enabled
    
    def set_voice(self, voice: str):
        """Change the voice."""
        self.voice = voice
    
    def greet(self):
        """Speak a greeting to start the conversation."""
        greeting = "Hello! I'm your voice assistant. How can I help you today?"
        print(f"Agent: {greeting}\n")
        self.speak(greeting)


# Available neural voices for conversation
CONVERSATION_VOICES = {
    "joanna": "Joanna",      # US Female (default)
    "matthew": "Matthew",    # US Male
    "amy": "Amy",            # British Female
    "brian": "Brian",        # British Male
    "emma": "Emma",          # British Female
    "olivia": "Olivia",      # Australian Female
    "aria": "Aria",          # New Zealand Female
}


def main():
    """Run the voice conversation agent."""
    print("Voice Conversation Agent")
    print("=" * 40)
    print("Type your messages - I'll speak my responses!")
    print("Commands:")
    print("  mute     - Toggle voice on/off")
    print("  voice X  - Change voice (joanna, matthew, amy, brian, emma)")
    print("  quit     - Exit")
    print("Tip: You can paste multi-line prompts!\n")
    
    try:
        print("Initializing Amazon Polly client...")
        agent = VoiceConversationAgent()
        print("✓ Ready! Starting conversation...\n")
    except NoCredentialsError:
        print("\nError: AWS credentials not configured")
        print("\nTroubleshooting:")
        print("  1. Run: aws configure")
        print("  2. Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return
    except Exception as e:
        print(f"\nError initializing Polly: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify AWS credentials have Polly access (polly:SynthesizeSpeech)")
        print("  2. Check the AWS region supports Polly neural voices")
        return
    
    # Start with a greeting
    agent.greet()
    
    while True:
        user_input = get_multiline_input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            farewell = "Goodbye! It was nice talking with you."
            print(f"\nAgent: {farewell}")
            agent.speak(farewell)
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == "mute":
            enabled = agent.toggle_voice()
            status = "enabled" if enabled else "disabled"
            print(f"Voice {status}\n")
            continue
        
        if user_input.lower().startswith("voice "):
            voice_name = user_input[6:].strip().lower()
            if voice_name in CONVERSATION_VOICES:
                agent.set_voice(CONVERSATION_VOICES[voice_name])
                print(f"Voice changed to {voice_name}")
                agent.speak("Hello, this is my new voice.")
                print()
            else:
                print(f"Unknown voice: {voice_name}")
                print(f"Available: {', '.join(CONVERSATION_VOICES.keys())}\n")
            continue
        
        try:
            agent.chat(user_input)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
