# Tau2 Purple Agent

A purple agent for the [τ²-Bench](https://github.com/sierra-research/tau2-bench) benchmark on the [AgentBeats](https://agentbeats.dev) platform.

Competes in customer service tasks across **airline**, **retail**, and **telecom** domains, using LLM reasoning and tool calls to resolve user requests while following domain policies.

## How it works

The agent exposes an A2A endpoint. On each evaluation task:

1. The green agent sends a system prompt containing the domain policy, available tools, and the conversation so far.
2. The purple agent calls an LLM (OpenAI-compatible) and responds with a JSON action — either a tool call or a direct response to the user.
3. This loop continues until the task is resolved or the step limit is reached.

## Project Structure

```
src/
├─ agent.py       # LLM-powered agent logic
├─ executor.py    # A2A request handling
├─ messenger.py   # A2A messaging utilities
└─ server.py      # A2A server setup and agent card
tests/
└─ test_agent.py  # A2A conformance tests
Dockerfile
pyproject.toml
```

## Running Locally

```bash
# Install dependencies
uv sync

# Set your API key (supports any OpenAI-compatible endpoint)
export OPENAI_API_KEY=sk-...
# Optional: use a custom base URL (e.g. proxy)
export OPENAI_API_BASE=...
# Optional: choose model (default: gpt-4o-mini)
export OPENAI_MODEL=...

# Start the agent
uv run src/server.py
```

The server starts on port 9009.

## Running with Docker

```bash
docker build -t tau2-purple-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=sk-... tau2-purple-agent
```


## Testing

```bash
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```

## Publishing

The GitHub Actions workflow automatically builds, tests, and publishes the Docker image to GHCR on push to `main`:

```
ghcr.io/paulrychkov/agent-beats-tau2:latest
```

Make the package public in: GitHub → Your Profile → Packages → agent-beats-tau2 → Package Settings → Make Public.

## AgentBeats Registration

1. Go to [agentbeats.dev](https://agentbeats.dev) and click **Register Agent**
2. Select **Purple**, fill in the Docker image `ghcr.io/paulrychkov/agent-beats-tau2:latest`
3. Submit an assessment via **Quick Submit** on the [τ²-Bench leaderboard](https://agentbeats.dev/agentbeater/tau2-bench)
