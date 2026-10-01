"""
Voice Tool Agent - Real-Time Voice with Tools

A voice-enabled agent that can use tools during real-time conversations.
Demonstrates how to combine bidirectional streaming with function tools.

Prerequisites:
    brew install portaudio              # macOS — provides the C library PyAudio needs
    pip install -r requirements.txt     # installs strands-agents[bidi,bidi-io,bidi-pyaudio] + pins
    # Enable Nova Sonic model access in Amazon Bedrock console
    # Python 3.12+ (Nova Sonic's experimental AWS SDK client requires it)

Learning objectives:
- Understand how tools work with voice agents
- See concurrent tool execution during conversations
- Learn to design voice-friendly tool responses

Note: Use headphones to prevent audio feedback loops.
Press Ctrl+C to exit the conversation.
"""

import ast
import asyncio
import operator
import sys
from datetime import datetime

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


# Allowlist for the calculator: only arithmetic on numbers. The tool input is
# derived from user speech via the model, so it is untrusted. We evaluate it by
# walking a parsed AST and permitting only numeric literals and basic operators
# — never eval(). Names, calls, and attribute access are rejected, so untrusted
# input cannot reach arbitrary code.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arithmetic(expression: str) -> float:
    """Evaluate a numbers-and-operators expression without eval().

    Walks the parsed AST and permits only numeric literals and basic
    arithmetic. Rejects names, calls, and attribute access, so untrusted
    input cannot reach arbitrary code.

    Args:
        expression: A math expression such as "25 * 4" or "100 / 5".

    Raises:
        ValueError: If the expression contains anything but numbers and
            the allowed arithmetic operators.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            # Bound exponents so "2 ** 999999999" can't hang the agent.
            if isinstance(node.op, ast.Pow):
                exponent = _eval(node.right)
                if abs(exponent) > 100:
                    raise ValueError("exponent too large")
                return operator.pow(_eval(node.left), exponent)
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expression, mode="eval"))


# Define voice-friendly tools. Each tool prints its inputs and result so the
# viewer can see which tool the voice agent decided to call (otherwise tool
# use is invisible because the agent speaks the assembled answer).
@tool
def get_current_time() -> str:
    """Get the current time.
    
    Returns:
        The current time in a speakable format
    """
    now = datetime.now()
    output = now.strftime("It's %I:%M %p")
    print(f"  [tool] get_current_time() -> {output}")
    return output


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: City name or location
    """
    print(f"  [tool] get_weather(location={location!r})")
    # Simulated weather data (replace with real API in production)
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 58, "condition": "cloudy with a chance of rain"},
        "tokyo": {"temp": 68, "condition": "clear and pleasant"},
        "san francisco": {"temp": 65, "condition": "foggy"},
        "seattle": {"temp": 55, "condition": "rainy"},
        "miami": {"temp": 85, "condition": "hot and humid"},
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        data = weather_data[location_lower]
        output = f"The weather in {location} is {data['temp']} degrees and {data['condition']}."
    else:
        output = f"I don't have weather data for {location}."
    print(f"         -> {output}")
    return output


@tool
def calculate(expression: str) -> str:
    """Perform a calculation.
    
    Args:
        expression: A math expression like "25 times 4" or "100 divided by 5"
    """
    print(f"  [tool] calculate(expression={expression!r})")
    # Convert spoken math to symbols
    expression = expression.lower()
    expression = expression.replace("plus", "+")
    expression = expression.replace("minus", "-")
    expression = expression.replace("times", "*")
    expression = expression.replace("multiplied by", "*")
    expression = expression.replace("divided by", "/")
    expression = expression.replace("over", "/")
    
    try:
        # Injection-safe: _safe_arithmetic() walks a parsed AST and allows
        # only numbers and arithmetic operators. It never calls eval(), so
        # untrusted input from speech can't reach arbitrary code.
        result = _safe_arithmetic(expression)
        output = f"The answer is {result}"
    except Exception:
        output = "I couldn't calculate that. Please try a simpler expression."
    print(f"         -> {output}")
    return output


@tool
def set_reminder(task: str, minutes: int) -> str:
    """Set a reminder for a task.
    
    Args:
        task: What to be reminded about
        minutes: How many minutes from now
    """
    print(f"  [tool] set_reminder(task={task!r}, minutes={minutes})")
    # In a real app, this would schedule an actual reminder
    output = f"I've set a reminder for '{task}' in {minutes} minutes."
    print(f"         -> {output}")
    return output


SYSTEM_PROMPT = """You are a helpful voice assistant with access to tools.

Available tools:
- get_current_time: Tell the current time
- get_weather: Get weather for a city
- calculate: Do math calculations
- set_reminder: Set reminders

Guidelines:
- Keep responses concise and natural for speech
- Use tools when the user asks about time, weather, math, or reminders
- Speak the tool results naturally, don't just read them verbatim
- If asked to stop, use the stop_conversation tool

Example interactions:
- "What time is it?" → Use get_current_time
- "What's the weather in Tokyo?" → Use get_weather
- "What's 25 times 4?" → Use calculate
- "Remind me to call mom in 30 minutes" → Use set_reminder"""


async def run_voice_tool_agent():
    """Run the voice agent with tools."""
    print("Voice Tool Agent (Real-Time)")
    print("=" * 40)
    print("I can help with: time, weather, math, and reminders")
    print("⚠️  Use HEADPHONES to prevent audio feedback loops.")
    print("   Without headphones, the agent will hear itself and interrupt constantly.")
    print("Say 'stop conversation' or press Ctrl+C to exit.\n")
    
    print("Try saying:")
    print("  - 'What time is it?'")
    print("  - 'What's the weather in New York?'")
    print("  - 'What's 25 times 4?'")
    print("  - 'Set a reminder to take a break in 30 minutes'\n")
    
    # Create bidirectional streaming model (Nova Sonic via Bedrock).
    # `params` is passed straight through as Nova Sonic's sessionStart fields.
    # LOW endpointing sensitivity reduces false interruptions from background
    # noise and speaker feedback. Use HIGH for quieter environments with headphones.
    model = BedrockNovaSonicModel(
        model_id=NOVA_SONIC_MODEL_ID,
        region=nova_sonic_region(),
        params={"turnDetectionConfiguration": {"endpointingSensitivity": "LOW"}},
    )
    
    # Create agent with tools
    agent = BidiAgent(
        model=model,
        tools=[get_current_time, get_weather, calculate, set_reminder, stop_conversation],
        system_prompt=SYSTEM_PROMPT
    )
    
    # Microphone in, speakers out. AudioIO's output stream also renders live
    # transcripts and the name of each tool the agent calls in the terminal, so
    # no separate text output is needed. Those transcripts include the user's
    # own speech: treat them as potentially sensitive (PII) and redact before
    # logging in a real service.
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
    """Entry point for the voice tool agent."""
    try:
        asyncio.run(run_voice_tool_agent())
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
