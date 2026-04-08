import json
import logging
import os
from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIRST_MESSAGE_SEPARATOR = "Now here are the user messages:"
TOOLS_SECTION_START = "Here's a list of tools you can use"


class Agent:
    def __init__(self):
        self.history: list[dict] = []
        self.system_prompt: str | None = None
        self.tools: list[dict] = []
        self.pending_tool_call_id: str | None = None
        self.turn = 0
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def _parse_first_message(self, text: str) -> tuple[str, str]:
        if FIRST_MESSAGE_SEPARATOR in text:
            system_part, user_part = text.split(FIRST_MESSAGE_SEPARATOR, 1)
            return system_part.strip(), user_part.strip()
        return "", text

    def _extract_tools(self, system_text: str) -> list[dict]:
        """Parse OpenAI-compatible tool definitions from the system prompt text."""
        try:
            start = system_text.find('[', system_text.find(TOOLS_SECTION_START))
            if start == -1:
                return []
            decoder = json.JSONDecoder()
            tools, _ = decoder.raw_decode(system_text, start)
            return [t for t in tools if t.get('function', {}).get('name') != 'respond']
        except Exception as e:
            logger.warning("Failed to extract tools: %s", e)
            return []

    def _is_tool_result(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith(('{', '[')):
            return False
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        self.turn += 1
        input_text = get_message_text(message)

        if self.system_prompt is None:
            system_ctx, user_content = self._parse_first_message(input_text)
            self.system_prompt = system_ctx
            self.tools = self._extract_tools(system_ctx)
            if user_content:
                self.history.append({"role": "user", "content": user_content})
            logger.info("[turn %d] INIT tools=%d | user: %s", self.turn, len(self.tools), user_content[:200])
        else:
            if self.pending_tool_call_id:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": self.pending_tool_call_id,
                    "content": input_text,
                })
                self.pending_tool_call_id = None
                logger.info("[turn %d] TOOL: %s", self.turn, input_text[:200])
            else:
                self.history.append({"role": "user", "content": input_text})
                logger.info("[turn %d] USER: %s", self.turn, input_text[:200])

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        try:
            kwargs: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.history,
                ],
                "temperature": 0.0,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"
                kwargs["parallel_tool_calls"] = False

            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                self.pending_tool_call_id = tool_call.id
                self.history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls],
                })
                reply = json.dumps({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                })
            else:
                content = choice.message.content or ""
                self.history.append({"role": "assistant", "content": content})
                reply = json.dumps({
                    "name": "respond",
                    "arguments": {"content": content},
                })

            logger.info("[turn %d] AGENT: %s", self.turn, reply[:300])

        except Exception as e:
            logger.error("[turn %d] Error: %s", self.turn, e)
            reply = json.dumps({
                "name": "respond",
                "arguments": {"content": "I'm sorry, I encountered an error. Please try again."},
            })

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )
