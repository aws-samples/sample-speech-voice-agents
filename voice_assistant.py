"""
Voice Assistant - Real-Time Bidirectional Streaming

A voice-enabled agent that has real-time conversations using
Strands' bidirectional streaming (strands.bidi) with Nova Sonic.

Prerequisites:
    brew install portaudio              # macOS — provides the C library PyAudio needs
    pip install -r requirements.txt     # installs strands-agents[bidi,bidi-io,bidi-pyaudio] + pins
    # Enable Nova Sonic model access in Amazon Bedrock console
    # Python 3.12+ (Nova Sonic's experimental AWS SDK client requires it)

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

# This agent needs Strands' bidi stack, which comes from
# strands-agents[bidi,bidi-io,bidi-pyaudio] in requirements.txt. If the import
# fails, the user either skipped the install or hit PyAudio build issues on
# Python 3.13. Either way, point them at the fix.
try:
    from strands import LocalAgent, ToolContext, tool
    from strands.bidi import BidiAgent
    from strands.bidi.io import AudioIO
    from strands.bidi.models import BedrockNovaSonicModel
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


# Nova Sonic 2 is the only model this agent supports; the bidi API requires an
# explicit model id (no default), so name it once here.
NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"

# Nova Sonic is only available in a few regions. Start from the shared region
# (.env AWS_REGION, then the AWS profile) and fall back to us-east-1 with a
# notice if that region does not host the model, instead of failing on connect.
NOVA_SONIC_REGIONS = ("us-east-1", "us-west-2", "eu-north-1", "ap-northeast-1")


def nova_sonic_region() -> str:
    report_aws_target(NOVA_SONIC_MODEL_ID)
    region = get_region()
    if region in NOVA_SONIC_REGIONS:
        return region
    print(f"Note: Nova Sonic is not available in {region}; using us-east-1.")
    print(f"      Supported regions: {', '.join(NOVA_SONIC_REGIONS)}\n")
    return "us-east-1"


@tool(context=True)
def stop_conversation(tool_context: ToolContext[LocalAgent]) -> str:
    """End the conversation when the user asks to stop.

    The bidi loop checks for cancellation after each tool group completes, so
    agent.run() returns on its own once this tool has run.
    """
    tool_context.agent.cancel()
    return "Ending the conversation. Goodbye!"


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
    
    # Create bidirectional streaming model (Nova Sonic via Bedrock).
    # `params` is passed straight through as Nova Sonic's sessionStart fields.
    # LOW endpointing sensitivity reduces false interruptions from background
    # noise and speaker feedback. Use HIGH for quieter environments with headphones.
    model = BedrockNovaSonicModel(
        model_id=NOVA_SONIC_MODEL_ID,
        region=nova_sonic_region(),
        params={"turnDetectionConfiguration": {"endpointingSensitivity": "LOW"}},
    )
    
    # Create voice-enabled agent with stop tool
    agent = BidiAgent(
        model=model,
        tools=[stop_conversation],
        system_prompt=SYSTEM_PROMPT
    )
    
    # Microphone in, speakers out. AudioIO's output stream also renders live
    # transcripts and tool calls in the terminal (user speech in shaded ">"
    # blocks, assistant speech as plain text), so no separate text output is
    # needed. Those transcripts include the user's own speech: treat them as
    # potentially sensitive (PII) and redact before logging in a real service.
    audio_io = AudioIO()
    
    try:
        print("Connecting to Nova Sonic... (initial connection takes 5-15 seconds)")
        print("When the 'Speak…' prompt appears, the assistant is listening.\n")
        
        # Runs until stop_conversation cancels the agent or Ctrl+C.
        await agent.run(
            inputs=[audio_io.input()],
            outputs=[audio_io.output()],
        )
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
        # run() stops the agent on exit; calling stop() again is a safe no-op
        # and covers the path where run() never started.
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
