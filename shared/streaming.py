"""
Streaming callback handler for visible agent reasoning.

The Strands SDK fires events to a `callback_handler` while an agent runs:
text-token deltas, tool-use starts, reasoning text, lifecycle markers. The
default `PrintingCallbackHandler` streams everything but its formatting is
minimal. This module provides a slightly nicer alternative that:

- Streams model text token-by-token as it arrives, so the user can watch the
  agent think instead of waiting for the whole response.
- Announces tool invocations the moment the model decides to call one, so
  the user knows whether the agent worked something out on its own or
  delegated to a tool.
- Says so when Bedrock throttles a call and Strands is waiting to retry, so a
  rate-limited agent doesn't look like a hung one.
- Leaves the tool's *outcome* (arguments and return value) to be printed by
  the tool function itself. That keeps tool tracing close to the code that
  actually does the work.

Note: this handler writes the model's raw output straight to stdout for
local, interactive use. That output can include user input or retrieved
context (potentially sensitive/PII). A production service should not stream
unfiltered output to shared logs -- filter/redact first, or use a
non-streaming handler that captures only what you intend to persist.

Usage:

    from shared.streaming import StreamingCallbackHandler

    agent = Agent(
        system_prompt="...",
        tools=[...],
        callback_handler=StreamingCallbackHandler(),
    )

See the Strands docs on callback handlers:
https://strandsagents.com/docs/user-guide/concepts/streaming/callback-handlers/
"""

import sys
from typing import Any


class StreamingCallbackHandler:
    """Stream model output and announce tool calls inline."""

    def __init__(self, tool_prefix: str = "  [tool] ") -> None:
        """Initialize the handler.

        Args:
            tool_prefix: String prepended to each tool-call announcement.
        """
        self._tool_prefix = tool_prefix

        # State carried across the many `__call__` invocations a single agent
        # run produces. Reset between turns so the next prompt starts clean.
        self._streaming_text = False
        self._announced_tools: set[str] = set()

    def reset(self) -> None:
        """Clear per-turn state. Call between agent invocations if reusing."""
        self._streaming_text = False
        self._announced_tools.clear()

    def __call__(self, **kwargs: Any) -> None:
        # 1. Tool-use start — fired when the model decides to call a tool.
        event = kwargs.get("event") or {}
        tool_use = (
            event.get("contentBlockStart", {}).get("start", {}).get("toolUse")
        )
        if tool_use:
            self._announce_tool(tool_use)
            return

        # 2. Model text deltas — stream straight to stdout.
        data = kwargs.get("data")
        if data:
            self._streaming_text = True
            sys.stdout.write(data)
            sys.stdout.flush()
            return

        # 3. Reasoning text — some models emit a separate reasoning channel.
        reasoning_text = kwargs.get("reasoningText")
        if reasoning_text:
            self._streaming_text = True
            sys.stdout.write(reasoning_text)
            sys.stdout.flush()
            return

        # 4. Throttling — Bedrock returned ThrottlingException and Strands is
        # backing off before it retries (4s, 8s, 16s, 32s, 64s by default).
        # Without this line a throttled call looks exactly like a hang. Bursts
        # of tool calls (one model call per tool call) are the usual trigger.
        delay = kwargs.get("event_loop_throttled_delay")
        if delay:
            if self._streaming_text:
                sys.stdout.write("\n")
                self._streaming_text = False
            sys.stdout.write(
                f"{self._tool_prefix}(Bedrock is rate-limiting this model; "
                f"retrying in {delay}s...)\n"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------ helpers

    def _announce_tool(self, tool_use: dict[str, Any]) -> None:
        tool_id = tool_use.get("toolUseId")
        tool_name = tool_use.get("name", "<unknown>")
        if not tool_id or tool_id in self._announced_tools:
            return
        self._announced_tools.add(tool_id)

        # If we were mid-stream of model text, close that line first so the
        # tool announcement doesn't run into the prose.
        if self._streaming_text:
            sys.stdout.write("\n")
            self._streaming_text = False

        sys.stdout.write(f"\n{self._tool_prefix}{tool_name}\n")
        sys.stdout.flush()
