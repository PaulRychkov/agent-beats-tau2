import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="tau2-customer-service",
        name="Customer Service Agent",
        description=(
            "Handles customer service tasks across airline, retail, and telecom domains. "
            "Uses available tools to look up information, manage bookings, process requests, "
            "and resolves issues following domain policies."
        ),
        tags=["customer-service", "tau2", "tool-use"],
        examples=[
            "Help me change my flight.",
            "I need to return a product I bought.",
            "Why is my internet not working?",
        ],
    )

    agent_card = AgentCard(
        name="Tau2 Purple Agent",
        description=(
            "A purple agent for the τ²-Bench benchmark. Competes in customer service tasks "
            "across airline, retail, and telecom domains using LLM reasoning and tool calls."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
