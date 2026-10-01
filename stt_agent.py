"""
Speech-to-Text Agent - Amazon Transcribe Streaming Integration

An agent that processes local audio files by streaming them to Amazon Transcribe
and then reasoning about the content. No S3 bucket required.

Prerequisites:
    pip install -r requirements.txt
    pip install --no-deps "amazon-transcribe==0.6.4"

Learning objectives:
- Understand speech-to-text integration with agents
- See how to use Amazon Transcribe Streaming for audio processing
- Learn about real-time audio transcription workflows

Note: This uses Transcribe Streaming which accepts audio bytes directly
without needing S3. For long files or async batch processing, see the
Transcribe batch API (requires S3).
"""

import sys
import os
import asyncio
import wave
from pathlib import Path
from shared.model import get_model, get_region
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent

import boto3
from botocore.exceptions import ClientError

try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
except ImportError:
    # amazon-transcribe pins an older awscrt that conflicts with the bidi
    # stack in requirements.txt. The fix is the two-step install:
    print("Error: amazon-transcribe streaming SDK not installed.")
    print()
    print("From this folder, run:")
    print()
    print("    pip install -r requirements.txt")
    print('    pip install --no-deps "amazon-transcribe==0.6.4"')
    print()
    print("The --no-deps flag is intentional — amazon-transcribe needs to")
    print("ride on the awscrt version already pulled in by the bidi stack.")
    sys.exit(1)


# Where the `sample` command writes its synthesized audio. Keeping the
# files in the lab folder lets the user play them back with afplay/ffplay
# to confirm what was spoken before transcription.
SAMPLE_DIR = Path(__file__).parent / "samples"

# Default phrase used by the `sample` command when the user doesn't supply
# their own text. ~50 words, mentions concepts the lab has already
# introduced so the agent's summary is easy to evaluate.
DEFAULT_SAMPLE_PHRASE = (
    "Amazon Transcribe Streaming converts spoken audio into text in real time, "
    "without requiring an S3 bucket. It uses an HTTP/2 connection and event "
    "streams to deliver partial transcripts as the audio arrives. Combined "
    "with a language model, you can build agents that listen, understand, "
    "and reason about what people say."
)


SYSTEM_PROMPT = """You are a helpful assistant that processes transcribed audio.

When given a transcript:
1. Summarize the key points if it's long
2. Answer any questions about the content
3. Identify speakers if multiple are present
4. Note any unclear or potentially misheard words

Be helpful and accurate in your analysis."""


class TranscriptCollector(TranscriptResultStreamHandler):
    """Collects finalized transcript segments during streaming."""
    
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.transcript_parts = []
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript_parts.append(alt.transcript)
    
    def get_full_transcript(self) -> str:
        return " ".join(self.transcript_parts).strip()


class STTAgent:
    """Agent that processes local audio files via Amazon Transcribe Streaming."""
    
    # Supported audio formats and their sample rates
    SUPPORTED_FORMATS = {
        'wav': 'pcm',
        'flac': 'flac',
        'ogg': 'ogg-opus',
    }
    
    def __init__(self, region: str | None = None, sample_rate: int = 16000):
        """Initialize the STT agent.
        
        Args:
            region: AWS region for Transcribe Streaming
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.region = region or get_region()
        self.sample_rate = sample_rate
        self.client = TranscribeStreamingClient(region=self.region)
        # Polly client is created lazily — only when `sample` is invoked.
        # Keeps the import-time cost of STTAgent small for users who just
        # want to transcribe existing files.
        self._polly = None
        # Streaming handler so the user can watch the analysis agent reason
        # over the transcript token-by-token.
        self.stream_handler = StreamingCallbackHandler()
        self.agent = Agent(
            model=get_model(),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=self.stream_handler,
        )

    def generate_sample(self, phrase: str = DEFAULT_SAMPLE_PHRASE,
                        voice_id: str = "Joanna",
                        filename: str = "sample.wav") -> Path:
        """Synthesize a phrase with Amazon Polly and save it as a WAV file.

        Polly's ``pcm`` output is raw 16-bit signed little-endian linear PCM
        at the requested sample rate — exactly what Transcribe Streaming
        expects when the encoding is ``pcm`` and the container is WAV. We
        wrap the bytes in a 44-byte RIFF header using the stdlib ``wave``
        module so the file is recognizable by audio players too.

        Args:
            phrase: Text to synthesize.
            voice_id: Polly neural voice ID (default ``Joanna``).
            filename: Output filename inside ``samples/``.

        Returns:
            The path the wav was written to.
        """
        if self._polly is None:
            print(f"          boto3.client('polly', region_name={self.region!r})")
            self._polly = boto3.client('polly', region_name=self.region)
            print("          -> Polly client ready")

        SAMPLE_DIR.mkdir(exist_ok=True)
        path = SAMPLE_DIR / filename

        print(f"          polly.synthesize_speech(voice={voice_id!r}, format='pcm', rate={self.sample_rate})")
        try:
            response = self._polly.synthesize_speech(
                Text=phrase,
                OutputFormat='pcm',
                SampleRate=str(self.sample_rate),
                VoiceId=voice_id,
                Engine='neural',
            )
            pcm_bytes = response['AudioStream'].read()
        except ClientError as e:
            # Surface the actual failure (likely missing polly:SynthesizeSpeech
            # permission or no model access) rather than letting it propagate
            # as a confusing wave-module error later.
            raise RuntimeError(f"Polly call failed: {e}") from e

        print(f"          -> {len(pcm_bytes)} bytes of raw PCM")

        # Wrap raw PCM in a WAV container so the file works with any player
        # and so Transcribe Streaming reads it as `encoding=pcm`.
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)        # Polly returns mono
            wav.setsampwidth(2)        # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm_bytes)

        print(f"  [saved] {path.relative_to(path.parent.parent)}")
        return path
    
    async def _stream_file(self, audio_file: str, language: str = "en-US") -> str:
        """Stream a local audio file to Transcribe and return the full transcript."""
        # Determine encoding from file extension
        ext = audio_file.split('.')[-1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{ext}'. Transcribe Streaming requires: "
                f"{', '.join(self.SUPPORTED_FORMATS.keys())}. "
                f"Convert MP3 files to WAV using: ffmpeg -i input.mp3 -ar 16000 output.wav"
            )
        
        encoding = self.SUPPORTED_FORMATS[ext]
        
        # Start streaming transcription
        stream = await self.client.start_stream_transcription(
            language_code=language,
            media_sample_rate_hz=self.sample_rate,
            media_encoding=encoding,
        )
        
        # Read audio file and send as chunks
        async def write_chunks():
            chunk_size = 1024 * 8  # 8KB chunks
            with open(audio_file, 'rb') as f:
                while chunk := f.read(chunk_size):
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()
        
        # Collect transcripts as they arrive
        handler = TranscriptCollector(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())
        
        return handler.get_full_transcript()
    
    def transcribe_audio(self, audio_file: str, language: str = "en-US") -> str:
        """Transcribe a local audio file.
        
        Args:
            audio_file: Path to local audio file (wav, flac, or ogg)
            language: Language code (default: en-US)
            
        Returns:
            Transcribed text
        """
        print(f"Transcribing {audio_file}...")
        transcript = asyncio.run(self._stream_file(audio_file, language))
        print("✓ Transcription complete")
        return transcript
    
    def process_audio(self, audio_file: str, question: str = None) -> None:
        """Transcribe audio and stream the agent's analysis.
        
        Args:
            audio_file: Path to audio file
            question: Optional question about the transcript
        """
        transcript = self.transcribe_audio(audio_file)
        # NOTE: prints the transcript to stdout for local use; the audio is the
        # user's own. Treat transcripts as potentially sensitive (PII) and
        # filter/redact before logging in a production service.
        print(f"\nTranscript: {transcript}\n")
        
        # Build prompt for agent
        if question:
            prompt = f'Here is a transcript from an audio file:\n\n"{transcript}"\n\nUser question: {question}\n\nPlease answer the question based on the transcript.'
        else:
            prompt = f'Here is a transcript from an audio file:\n\n"{transcript}"\n\nPlease summarize the key points and any notable information.'

        self.stream_handler.reset()
        print("Agent: ", end="", flush=True)
        self.agent(prompt)
        print()


def main():
    """Run the STT agent interactively."""
    print("Speech-to-Text Agent (Amazon Transcribe Streaming)")
    print("=" * 40)
    print("I can transcribe local audio files and answer questions about them.")
    print("Supported formats: WAV, FLAC, OGG (no S3 bucket required!)")
    print("No audio file? Type 'sample' and I'll synthesize one with Polly.")
    print("Type 'quit' to exit")
    print("Tip: You can paste multi-line prompts!\n")

    try:
        print("Initializing Amazon Transcribe Streaming client...")
        stt_agent = STTAgent()
        print("✓ Ready! Use 'sample', 'transcribe <file>', or 'ask <file>' to get started.\n")
    except Exception as e:
        print(f"\nError initializing Transcribe client: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify AWS credentials are configured (run: aws configure)")
        print("  2. Confirm IAM permissions include: transcribe:StartStreamTranscription")
        print("     (and polly:SynthesizeSpeech if you plan to use 'sample')")
        print("  3. Check the AWS region supports Transcribe Streaming")
        return
    
    print("Commands:")
    print("  sample                 - Synthesize a sample phrase with Polly, then transcribe it")
    print("  sample <text>          - Same, but synthesize the text you provide")
    print("  transcribe <file.wav>  - Transcribe and summarize an audio file")
    print("  ask <file.wav>         - Transcribe, then ask follow-up questions")
    print()
    
    current_transcript = None
    
    while True:
        user_input = get_multiline_input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            if user_input.lower() == "sample" or user_input.lower().startswith("sample "):
                phrase = user_input[7:].strip() if len(user_input) > 6 else DEFAULT_SAMPLE_PHRASE
                if not phrase:
                    phrase = DEFAULT_SAMPLE_PHRASE
                sample_path = stt_agent.generate_sample(phrase=phrase)
                stt_agent.process_audio(str(sample_path))

            elif user_input.lower().startswith("transcribe "):
                audio_file = user_input[11:].strip()
                if not os.path.exists(audio_file):
                    print(f"File not found: {audio_file}\n")
                    continue
                
                stt_agent.process_audio(audio_file)
                
            elif user_input.lower().startswith("ask "):
                audio_file = user_input[4:].strip()
                if not os.path.exists(audio_file):
                    print(f"File not found: {audio_file}\n")
                    continue
                
                current_transcript = stt_agent.transcribe_audio(audio_file)
                # NOTE: prints the transcript to stdout for local use; the audio
                # is the user's own. Treat transcripts as potentially sensitive
                # (PII) and filter/redact before logging in a production service.
                print(f"\nTranscript: {current_transcript}\n")
                print("Now you can ask questions about this transcript.\n")
                
            elif current_transcript:
                prompt = f'Transcript: "{current_transcript}"\n\nQuestion: {user_input}'
                stt_agent.stream_handler.reset()
                print("\nAgent: ", end="", flush=True)
                stt_agent.agent(prompt)
                print()
                
            else:
                print("Please transcribe an audio file first, or type 'sample' to generate one.")
                print("Usage: sample, transcribe <file.wav>, or ask <file.wav>\n")
                
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
