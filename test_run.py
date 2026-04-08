import asyncio
import argparse
import json
import sys
import io
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


async def run_evaluation(green_url: str, purple_url: str, domain: str = "airline", num_tasks: int = 1):
    payload = json.dumps({
        "participants": {"agent": purple_url},
        "config": {"domain": domain, "num_tasks": num_tasks},
    })

    print(f"Green agent : {green_url}")
    print(f"Purple agent: {purple_url}")
    print(f"Domain      : {domain}  |  Tasks: {num_tasks}")
    print("-" * 50)

    async with httpx.AsyncClient(timeout=600) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=green_url)
        agent_card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=payload))],
            message_id=uuid4().hex,
        )

        async for event in client.send_message(msg):
            match event:
                case Message() as m:
                    for part in m.parts:
                        if hasattr(part.root, "text"):
                            print(part.root.text)
                case (task, update):
                    if update and hasattr(update, "status") and update.status.message:
                        for part in update.status.message.parts:
                            if hasattr(part.root, "text"):
                                print(f"[status] {part.root.text}")
                    if task.artifacts:
                        for artifact in task.artifacts:
                            for part in artifact.parts:
                                if hasattr(part.root, "text"):
                                    print("\n=== RESULT ===")
                                    print(part.root.text)
                                if hasattr(part.root, "data"):
                                    print("\n=== SCORE DATA ===")
                                    print(json.dumps(part.root.data, indent=2))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--green", default="http://localhost:8099")
    parser.add_argument("--purple", default="http://tau2-purple:9009/")
    parser.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom"])
    parser.add_argument("--tasks", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.green, args.purple, args.domain, args.tasks))
