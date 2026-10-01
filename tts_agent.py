"""
Text-to-Speech Agent - Amazon Polly Integration

An agent that generates spoken audio responses using Amazon Polly.
Streams audio directly to speakers without saving to disk.

Prerequisites:
    pip install -r requirements.txt

Learning objectives:
- Understand text-to-speech integration with agents
- See how to use Amazon Polly for speech synthesis
- Learn about voice selection and audio output options
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


SYSTEM_PROMPT = """You are a helpful assistant optimized for spoken responses.

Guidelines:
- Keep responses concise (1-3 sentences ideal for speech)
- Avoid special characters, code blocks, or formatting
- Use natural, conversational language
- Spell out abbreviations and acronyms
- Use simple sentence structures

When asked a question, provide a clear, spoken-friendly answer."""


# Available Polly neural voices
VOICES = {
    "joanna": {"id": "Joanna", "gender": "Female", "accent": "US English"},
    "matthew": {"id": "Matthew", "gender": "Male", "accent": "US English"},
    "amy": {"id": "Amy", "gender": "Female", "accent": "British English"},
    "brian": {"id": "Brian", "gender": "Male", "accent": "British English"},
    "emma": {"id": "Emma", "gender": "Female", "accent": "British English"},
    "ivy": {"id": "Ivy", "gender": "Female (child)", "accent": "US English"},
    "kendra": {"id": "Kendra", "gender": "Female", "accent": "US English"},
    "salli": {"id": "Salli", "gender": "Female", "accent": "US English"},
}


def _play_audio_stream(audio_bytes: bytes) -> None:
    """Play MP3 audio bytes through the system audio without saving to disk.
    
    Uses ffplay (bundled with ffmpeg) to stream audio from stdin.
    Falls back to afplay/mpg123 with temporary pipe on unsupported systems.
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


class TTSAgent:
    """Agent that speaks responses using Amazon Polly."""
    
    def __init__(self, voice: str = "joanna", region: str | None = None):
        """Initialize the TTS agent.
        
        Args:
            voice: Voice name (joanna, matthew, amy, brian, etc.)
            region: AWS region for Polly
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
        self.voice_id = VOICES.get(voice.lower(), VOICES["joanna"])["id"]
    
    def speak(self, text: str) -> bool:
        """Convert text to speech and play it immediately (no disk writes).
        
        Args:
            text: Text to convert to speech
            
        Returns:
            True if successful
        """
        try:
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=self.voice_id,
                Engine='neural'
            )
            
            # Stream the audio bytes directly to the player
            audio_bytes = response['AudioStream'].read()
            _play_audio_stream(audio_bytes)
            return True
            
        except ClientError as e:
            print(f"Polly error: {e}")
            return False
    
    def chat(self, user_input: str, speak: bool = True) -> str:
        """Stream the response and optionally speak it.

        Args:
            user_input: User's text input
            speak: Whether to play audio (default True)

        Returns:
            The agent's text response (assembled from the streamed tokens)
        """
        self.stream_handler.reset()
        print("Agent: ", end="", flush=True)
        response = self.agent(user_input)
        print()  # close the streamed line
        response_text = str(response)

        if speak:
            self.speak(response_text)

        return response_text
    
    def list_voices(self):
        """List available voices."""
        print("\nAvailable voices:")
        for name, info in VOICES.items():
            print(f"  {name}: {info['gender']}, {info['accent']}")
        print()


def main():
    """Run the TTS agent interactively."""
    print("Text-to-Speech Agent (Amazon Polly)")
    print("=" * 40)
    print("I'll speak my responses using Amazon Polly.")
    print("Type 'voices' to see available voices.")
    print("Type 'mute' to toggle audio on/off.")
    print("Type 'quit' to exit")
    print("Tip: You can paste multi-line prompts!\n")
    
    try:
        print("Initializing Amazon Polly client...")
        tts_agent = TTSAgent(voice="joanna")
        print("✓ Ready! Type a message to get a spoken response.\n")
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
    
    speak_enabled = True
    
    while True:
        user_input = get_multiline_input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "voices":
            tts_agent.list_voices()
            continue
        
        if user_input.lower() == "mute":
            speak_enabled = not speak_enabled
            status = "enabled" if speak_enabled else "disabled"
            print(f"Audio {status}\n")
            continue
        
        if user_input.lower().startswith("voice "):
            voice_name = user_input[6:].strip().lower()
            if voice_name in VOICES:
                tts_agent.voice_id = VOICES[voice_name]["id"]
                print(f"Voice changed to {voice_name}\n")
            else:
                print(f"Unknown voice: {voice_name}")
                tts_agent.list_voices()
            continue
        
        try:
            tts_agent.chat(user_input, speak=speak_enabled)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
