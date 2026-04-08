import json
import logging
import os
from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message


SYSTEM_PROMPT = """You are a customer service agent. You will receive tasks from a benchmark evaluator.

On the first turn you will receive a full system prompt describing your role, available tools, and the conversation so far.
On subsequent turns you will receive tool results or new user messages.

You MUST always respond with valid JSON in exactly this format:
{"name": "<function_name>", "arguments": {<arguments>}}

To respond to the user (instead of calling a tool), use:
{"name": "respond", "arguments": {"content": "<your message>"}}

Never include any text outside the JSON. Never call more than one tool at a time."""


class Agent:
    def __init__(self):
        self.history: list[dict] = []
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        self.history.append({"role": "user", "content": input_text})

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *self.history,
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            reply = response.choices[0].message.content or "{}"
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            reply = json.dumps({"name": "respond", "arguments": {"content": "I'm sorry, I encountered an error. Please try again."}})

        try:
            json.loads(reply)
        except json.JSONDecodeError:
            logging.warning(f"LLM returned invalid JSON: {reply[:200]}")
            reply = json.dumps({"name": "respond", "arguments": {"content": reply}})

        self.history.append({"role": "assistant", "content": reply})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )
