# Speech and Voice Agents

Agents that hold spoken conversations, combining speech-to-text, LLM reasoning, and text-to-speech so users can talk to them instead of typing.

This sample builds voice agents with the [Strands Agents SDK](https://strandsagents.com/), Amazon Nova Sonic, Amazon Polly, and Amazon Transcribe, and is based off of the [AWS Prescriptive Guidance - Speech and voice agents pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/speech-and-voice-agents.html).

## Table of Contents

- [Quick Start](#quick-start)
- [Real-Time Voice Agents](#real-time-voice-agents)
  - [How It Works](#how-it-works)
  - [Voice Assistant](#voice-assistant)
  - [Voice Agent with Tools](#voice-agent-with-tools)
  - [Model Providers](#model-providers)
- [Turn-Based Voice Agents](#turn-based-voice-agents)
  - [Text-to-Speech](#text-to-speech)
  - [Speech-to-Text](#speech-to-text)
  - [Conversational Voice](#conversational-voice)
- [AWS Implementation Patterns](#aws-implementation-patterns)
- [Reference](#reference)

## Quick Start

**Prerequisites:**
- Python 3.12+ for the real-time agents (Nova Sonic's experimental AWS SDK client requires it); 3.10+ for the turn-based agents
- An AWS account with Amazon Bedrock access
- AWS credentials configured (`aws configure`) with permission to invoke models on Bedrock
- A **microphone and headphones** (real-time agents) and an audio player such as `ffplay`, `afplay`, or `mpg123` (turn-based TTS)
- **PortAudio** for the real-time agents: `brew install portaudio` (macOS) / `sudo apt-get install portaudio19-dev` (Linux)
- **Amazon Nova Sonic** model access enabled for the real-time agents (available in `us-east-1`, `us-west-2`, `eu-north-1`, `ap-northeast-1`)

>**Note:** A headset is required to have a voice conversation or the agent will interrupt itself

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

brew install portaudio
pip install -r requirements.txt

# Point the sample at your AWS profile and region (loaded by shared/model.py)
cp .env.example .env
# Edit .env: set AWS_PROFILE and AWS_REGION. Optionally pin a model with STRANDS_MODEL_ID.
pip install --no-deps "amazon-transcribe==0.6.4"

# Real-time (bidirectional streaming — needs mic, headphones, Nova Sonic access)
python voice_assistant.py     # full-duplex spoken conversation
python voice_tool_agent.py    # spoken conversation that can call tools

# Turn-based (needs Polly / Transcribe access)
python tts_agent.py           # text in, spoken response (Amazon Polly)
python stt_agent.py           # audio file in, transcript + reasoning (Amazon Transcribe)
python voice_conversation.py  # multi-turn chat with spoken responses
```

**Try these exercises:**
1. **Have a conversation.** Experiment with the voice assistant using headphones and talk to the agent; interrupt it mid-sentence and watch it stop and listen.
2. **Give voice a tool.** Ask the voice-tool agent for something that triggers a tool (e.g., the current time) and hear it answer with real data.
3. **Change the voice.** Use the tts agent by typing `voices`, then `voice brian`, and compare how the same answer sounds.
4. **Transcribe and reason.** Point the stt agent at a local audio file and ask it to summarize what was said.

---

## Real-Time Voice Agents

The modern approach is **bidirectional streaming**: audio flows to and from the model continuously, so the agent listens and speaks at the same time and can be interrupted mid-sentence. These agents use [Amazon Nova Sonic](https://docs.aws.amazon.com/nova/latest/userguide/speech.html), a speech-to-speech model, through the Strands bidirectional streaming API (`strands.bidi`).

### How It Works

1. **Receives a voice query**: the user speaks to a phone, microphone, or embedded system, and speech-to-text converts the audio to text
2. **Integrates streaming and telephony context**: a streaming interface manages real-time audio I/O; in a contact-center context, telephony handles session routing, touch-tone (DTMF) input, and media transport
3. **Reasons through LLM stream context**: the query and any session metadata (caller ID, prior context) go to the LLM, which responds using multiturn memory for ongoing interactions
4. **Returns a voice response**: text-to-speech converts the response to audio and returns it through the voice channel

<img src="images/speech-voice-agents.png" width="600" alt="Diagram of a speech and voice agent: spoken audio is transcribed to text, an LLM reasons over it with session context, and the response is synthesized back to speech and returned through the voice channel." />


### Voice Assistant

The [voice assistant](voice_assistant.py) is a full-duplex spoken conversation. A `BidiAgent` wraps a `BedrockNovaSonicModel`, and `AudioIO` wires the microphone and speakers to the stream and renders live transcripts in the terminal:

```python
from strands.bidi import BidiAgent
from strands.bidi.io import AudioIO
from strands.bidi.models import BedrockNovaSonicModel

model = BedrockNovaSonicModel(model_id="amazon.nova-2-sonic-v1:0", region="us-east-1")
agent = BidiAgent(model=model, system_prompt="You are a helpful voice assistant...")
audio_io = AudioIO()
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
```

> **Note** The system prompt steers the agent to keep responses short and conversational. Endpointing sensitivity is set to `LOW` (via the model's `params`, which map to Nova Sonic's `sessionStart` fields) to reduce false interruptions from background noise. Saying "stop conversation" runs a small `@tool(context=True)` that calls `agent.cancel()`, which ends `agent.run()` cleanly.

### Voice Agent with Tools

The [voice tool agent](voice_tool_agent.py) is the same real-time setup plus `@tool` functions, so the voice agent can *act*, i.e. look up the time or perform a calculation, and speak the result. Tools attach exactly as they did in [Sample 02](https://github.com/aws-samples/sample-tool-based-agents-functions).

### Model Providers

The sample uses Amazon Nova Sonic as the AWS-native speech-to-speech model. The Strands bidirectional API is provider-agnostic — it also supports other real-time providers — but everything here targets Nova Sonic on Amazon Bedrock.

---

## Turn-Based Voice Agents

Before speech-to-speech models, voice was a **pipeline**: convert speech to text, reason over the text, convert the answer back to speech. That pattern is still useful when you only need one direction (just speaking, or just listening) or want to mix and match services. These agents use [Amazon Polly](https://aws.amazon.com/polly/) for synthesis and [Amazon Transcribe](https://aws.amazon.com/transcribe/) for recognition.

### Text-to-Speech

The [TTS agent](tts_agent.py) takes typed input, reasons over it, and speaks the answer with Amazon Polly's neural voices — streaming the audio straight to your speakers without writing a file. Type `voices` to list the available voices and `voice <name>` to switch. The prompt is tuned for speech: short sentences, no code blocks, spelled-out acronyms.

### Speech-to-Text

The [STT agent](stt_agent.py) goes the other way: it streams a local audio file to Amazon Transcribe (the streaming API, so no S3 bucket is needed), then reasons over the transcript.

### Conversational Voice

The [voice conversation agent](voice_conversation.py) is a multi-turn assistant that keeps the conversation going across turns and speaks each response with Polly.

---

## AWS Implementation Patterns

| Pattern | Description | Reference |
|---------|-------------|-----------|
| Multi-agent voice assistant | Build a multi-agent voice assistant with Nova Sonic on Amazon Bedrock AgentCore | [Building a multi-agent voice assistant with Amazon Nova Sonic and Amazon Bedrock AgentCore](https://aws.amazon.com/blogs/machine-learning/building-a-multi-agent-voice-assistant-with-amazon-nova-sonic-and-amazon-bedrock-agentcore/) |
| Voice-driven AWS assistant | Create a voice-driven assistant for AWS tasks using Amazon Nova Sonic | [Building a voice-driven AWS assistant with Amazon Nova Sonic](https://aws.amazon.com/blogs/machine-learning/building-a-voice-driven-aws-assistant-with-amazon-nova-sonic/) |
| Text agent → voice assistant | Migrate an existing text agent to a voice assistant with Amazon Nova Sonic | [Migrating a text agent to a voice assistant with Amazon Nova 2 Sonic](https://aws.amazon.com/blogs/machine-learning/migrating-a-text-agent-to-a-voice-assistant-with-amazon-nova-2-sonic/) |
| Scalable voice agent design | Design scalable voice agents with Nova Sonic, multi-agent tools, and session segmentation | [Scalable voice agent design with Amazon Nova Sonic, multi-agent tools, and session segmentation](https://aws.amazon.com/blogs/machine-learning/scalable-voice-agent-design-with-amazon-nova-sonic-multi-agent-tools-and-session-segmentation/) |

## Reference

- [AWS Prescriptive Guidance - Speech and voice agents](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/speech-and-voice-agents.html)
- [Amazon Nova Sonic](https://docs.aws.amazon.com/nova/latest/userguide/speech.html)
- [Amazon Polly](https://aws.amazon.com/polly/)
- [Amazon Transcribe](https://aws.amazon.com/transcribe/)
- [Strands Agents Documentation](https://strandsagents.com/)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)

### The series

This sample is one of eleven, one per pattern in the [AWS Prescriptive Guidance on agentic AI patterns](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/). Each has a hands-on sample repository.

| # | Pattern | Sample |
|---|---|---|
| 01 | Basic Reasoning Agents | [sample-basic-reasoning-agents](https://github.com/aws-samples/sample-basic-reasoning-agents) |
| 02 | Tool-Based Agents (Functions) | [sample-tool-based-agents-functions](https://github.com/aws-samples/sample-tool-based-agents-functions) |
| 03 | Tool-Based Agents (Servers) | [sample-tool-based-agents-servers](https://github.com/aws-samples/sample-tool-based-agents-servers) |
| 04 | Computer-Use Agents | [sample-computer-use-agents](https://github.com/aws-samples/sample-computer-use-agents) |
| 05 | Coding Agents | [sample-coding-agents](https://github.com/aws-samples/sample-coding-agents) |
| 06 | Speech and Voice Agents | this repository |
| 07 | Workflow Orchestration Agents | [sample-workflow-orchestration-agent](https://github.com/aws-samples/sample-workflow-orchestration-agent) |
| 08 | Memory-Augmented Agents | [sample-memory-augmented-agents](https://github.com/aws-samples/sample-memory-augmented-agents) |
| 09 | Simulation and Test-Bed Agents | [sample-simulation-testbed-agents](https://github.com/aws-samples/sample-simulation-testbed-agents) |
| 10 | Observer and Monitoring Agents | [sample-observer-monitoring-agents](https://github.com/aws-samples/sample-observer-monitoring-agents) |
| 11 | Multi-Agent Collaboration | [sample-multi-agent-collaboration](https://github.com/aws-samples/sample-multi-agent-collaboration) |

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
