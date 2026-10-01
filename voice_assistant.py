"""
Voice Assistant - Real-Time Bidirectional Streaming

A voice-enabled agent that has real-time conversations using
Strands' experimental bidirectional streaming with Nova Sonic.

Prerequisites:
    brew install portaudio              # macOS — provides the C library PyAudio needs
    pip install -r requirements.txt     # installs strands-agents[bidi,bidi-io] + pins
    # Enable Nova Sonic model access in Amazon Bedrock console

Learning objectives:
- Understand real-time voice agent architecture
- See how bidirectional streaming enables natural conversations
- Learn about interruption handling and audio I/O

Note: Use headphones to prevent audio feedback loops.
Press Ctrl+C to exit the conversation.
"""

import asyncio
import sys

from shared.model import get_region, report_aws_target

# This agent needs Strands' experimental bidi stack, which comes from
# strands-agents[bidi,bidi-io] in requirements.txt. If the import fails, the
# user either skipped the install or hit PyAudio build issues on
# Python 3.13. Either way, point them at the fix.
try:
    from strands.experimental.bidi import BidiAgent, BidiAudioIO
    from strands.experimental.bidi.models import BidiNovaSonicModel
    from strands.experimental.bidi.tools import stop_conversation
    from strands.experimental.bidi.types.events import (
        BidiOutputEvent,
        BidiTranscriptStreamEvent,
    )
    from strands.experimental.bidi.types.io import BidiOutput
except ImportError as e:
    print("Error: Bidirectional streaming dependencies are not installed.")
    print(f"  ({e})")
    print()
    print("From this folder, run:")
    print()
    print("    brew install portaudio          # macOS, once")
    print("    pip install -r requirements.txt")
    print()
    print("If pip fails to build PyAudio, make sure PortAudio is installed at")
    print("the system level. PyAudio's source build on Python 3.13 also needs")
    print("version 0.2.14+ — requirements.txt pins that.")
    sys.exit(1)


# Nova Sonic is only available in a few regions. Start from the shared region
# (.env AWS_REGION, then the AWS profile) and fall back to us-east-1 with a
# notice if that region does not host the model, instead of failing on connect.
NOVA_SONIC_REGIONS = ("us-east-1", "us-west-2", "ap-northeast-1")


def nova_sonic_region() -> str:
    report_aws_target()
    region = get_region()
    if region in NOVA_SONIC_REGIONS:
        return region
    print(f"Note: Nova Sonic is not available in {region}; using us-east-1.")
    print(f"      Supported regions: {', '.join(NOVA_SONIC_REGIONS)}\n")
    return "us-east-1"


class FinalTranscriptOutput(BidiOutput):
    """Print only finalized transcripts (skip streaming previews)."""
    
    async def __call__(self, event: BidiOutputEvent) -> None:
        if isinstance(event, BidiTranscriptStreamEvent) and event["is_final"]:
            role = event["role"].capitalize()
            # NOTE: prints live transcripts (including the user's own speech) to
            # stdout for local use. Treat transcripts as potentially sensitive
            # (PII) and filter/redact before logging in a production service.
            print(f"{role}: {event['text']}")


SYSTEM_PROMPT = """You are a helpful voice assistant.

Guidelines:
- Keep responses concise and natural (1-3 sentences)
- Be conversational and friendly
- If asked to stop, use the stop_conversation tool
- Speak clearly and at a moderate pace

You can help with:
- Answering questions
- Having casual conversations
- Providing information and explanations"""


async def run_voice_assistant():
    """Run the real-time voice assistant."""
    print("Voice Assistant (Real-Time)")
    print("=" * 40)
    print("Speak naturally - I'm listening!")
    print("⚠️  Use HEADPHONES to prevent audio feedback loops.")
    print("   Without headphones, the agent will hear itself and interrupt constantly.")
    print("Say 'stop conversation' or press Ctrl+C to exit.\n")
    
    # Create bidirectional streaming model (Nova Sonic via Bedrock)
    # LOW endpointing sensitivity reduces false interruptions from background noise
    # and speaker feedback. Use HIGH for quieter environments with headphones.
    model = BidiNovaSonicModel(
        provider_config={
            "turn_detection": {
                "endpointingSensitivity": "LOW"
            }
        },
        client_config={"region": nova_sonic_region()}
    )
    
    # Create voice-enabled agent with stop tool
    agent = BidiAgent(
        model=model,
        tools=[stop_conversation],
        system_prompt=SYSTEM_PROMPT
    )
    
    # Setup audio I/O for microphone and speakers
    # Also add text output for finalized transcript visibility (no streaming previews)
    audio_io = BidiAudioIO()
    text_io = FinalTranscriptOutput()
    
    try:
        print("Connecting to Nova Sonic... (initial connection takes 5-15 seconds)")
        
        # Run the agent in a background task so we can detect when it's connected
        agent_task = asyncio.create_task(agent.run(
            inputs=[audio_io.input()],
            outputs=[audio_io.output(), text_io]  # Audio + final transcripts only
        ))
        
        # Poll for connection - agent._started becomes True once start() completes
        while not agent_task.done():
            if getattr(agent, '_started', False):
                print("✓ Connected! Start speaking - the assistant is listening.\n")
                print("(Transcript will appear below as you converse)\n")
                break
            await asyncio.sleep(0.1)
        
        # Wait for the agent loop to finish (or raise exception)
        await agent_task
    except asyncio.CancelledError:
        print("\nConversation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify Nova Sonic model access is enabled in the Bedrock console")
        print("     https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess")
        print("  2. Confirm IAM permissions include bedrock:InvokeModelWithBidirectionalStream")
        print(f"  3. Check region - Nova Sonic is only in {', '.join(NOVA_SONIC_REGIONS)}")
    finally:
        await agent.stop()
        print("Goodbye!")


def main():
    """Entry point for the voice assistant."""
    try:
        asyncio.run(run_voice_assistant())
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
