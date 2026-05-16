# Memanto + LangGraph Integration

## Permanent Memory for Your LangGraph Agents

This example demonstrates how to use **Memanto** as the long-term memory layer for a **LangGraph** agent workflow. The agent can remember information across completely separate sessions — something LangGraph's built-in state cannot do.

## The Problem

LangGraph is excellent for stateful agent workflows within a single session. But when a session ends, the graph state is lost. Your agent starts fresh every time, with no memory of past conversations, user preferences, or decisions.

Memanto solves this by providing persistent, queryable memory that survives across LangGraph sessions.

## Architecture

```
┌─────────────────────────────────────────────┐
│              LangGraph Workflow              │
│                                              │
│  Session 1 (separate)   Session 2            │
│  ┌──────────────┐      ┌──────────────┐     │
│  │  User says:  │      │  User asks:  │     │
│  │ "I'm vegan"  │      │ "What's my   │     │
│  └──────┬───────┘      │  diet pref?" │     │
│         │              └──────┬───────┘     │
│         ▼                     ▼              │
│  ┌──────────────┐      ┌──────────────┐     │
│  │ remember()   │      │  recall()    │     │
│  └──────┬───────┘      └──────┬───────┘     │
│         │                     │              │
└─────────┼─────────────────────┼──────────────┘
          │                     │
          ▼                     ▼
┌──────────────────────────────────────────────┐
│              Memanto Memory Store             │
│  (persistent — survives between sessions)    │
└──────────────────────────────────────────────┘
```

## Key Features

- **Cross-Session Recall**: Agent remembers information from past sessions
- **Typed Memories**: Store facts, preferences, instructions, and more using Memanto's memory categories
- **Confidence & Provenance**: Every memory carries metadata about when and how it was stored
- **No Indexing Delay**: Memories are searchable the instant they're stored
- **Minimal LangGraph State**: Keep your graph state lean — Memanto handles persistence

## Prerequisites

```bash
pip install memanto langgraph langchain-core
```

You'll also need a Memanto API key from [memanto.ai](https://memanto.ai) (free tier available).

## Quick Start

```python
from memanto_cli.client.sdk_client import SdkClient
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

# Initialize Memanto client
memanto = SdkClient(api_key="your-api-key")

# Create an agent in Memanto
agent = memanto.create_agent("CustomerSupportBot")

# Define the LangGraph node
def process_message(state: MessagesState) -> dict:
    """Process user message with Memanto memory."""
    user_msg = state["messages"][-1]

    # 1. RECALL: Check if we have relevant memories
    memory = memanto.recall(
        query=user_msg.content,
        namespace="customer_support",
        agent_id=agent["id"]
    )

    # 2. Build context from memory (cross-session!)
    context = memory.get("results", [])

    # 3. Generate response with memory context
    response = generate_response(user_msg.content, context)

    # 4. REMEMBER: Store what we learned
    memanto.remember(
        content=f"User asked: {user_msg.content}. We responded about X.",
        namespace="customer_support",
        agent_id=agent["id"],
        memory_type="fact"
    )

    return {"messages": [AIMessage(content=response)]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", process_message)
graph.set_entry_point("agent")
app = graph.compile()

# Session 1
result = app.invoke({"messages": [HumanMessage(content="I prefer vegetarian meals")]})

# Session 2 (new LangGraph run — but Memanto remembers!)
result = app.invoke({"messages": [HumanMessage(content="What did I tell you about food?")]})
# → Returns: "You mentioned you prefer vegetarian meals"
```

## Example: Customer Support Agent

### How Cross-Session Memory Works

**Session 1:**
```
User: I have a premium subscription
Agent: Great! I've noted your account type.

User: My shipping address is 123 Main St, Portland
Agent: Thanks, I've saved your shipping address.

User: I'm allergic to dairy
Agent: Noted! I'll remember this for future orders.
```

**Session 2 (hours later, fresh LangGraph state):**
```
User: What's my shipping address?
Agent: Your shipping address is 123 Main St, Portland.
       (Retrieved from Memanto — not from LangGraph state)

User: What allergies do I have?
Agent: You're allergic to dairy.
       (Cross-session recall!)

User: Place my usual order
Agent: I know you have a premium subscription.
       I'll place your usual order with the dairy-free option.
```

## Files

| File | Purpose |
|------|---------|
| `customer_support.py` | Complete LangGraph customer support agent with Memanto memory |
| `research_assistant.py` | Research assistant that stores and retrieves findings across sessions |
| `simple_demo.py` | Minimal example showing the core cross-session recall pattern |
| `README.md` | This file |

## How Memanto Solves LangGraph's Memory Gap

| LangGraph Limitation | Memanto Solution |
|---------------------|------------------|
| State lost when session ends | Persistent memory store |
| No cross-session context | Recall memories from any past session |
| Flat state only | Typed memories (facts, preferences, goals, etc.) |
| No search across conversations | Semantic recall query |
| No confidence/provenance | Every memory has metadata |

## License

MIT — Example code, free to use in your projects.
