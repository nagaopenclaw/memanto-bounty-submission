"""
Customer Support Agent — Memanto + LangGraph Integration
========================================================

A LangGraph workflow that uses Memanto as a persistent memory layer
for cross-session recall.

Core demonstration: The agent remembers user information across
completely separate LangGraph invocations (simulating separate chat
sessions hours or days apart).

Requirements:
    pip install memanto langgraph langchain-core

To run with real Memanto memory:
    export MEMANTO_API_KEY="your-key-here"
    python customer_support.py --live

To run in demo mode (simulated memory, no API key needed):
    python customer_support.py
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Memanto — try SDK client first, then fallback
try:
    from memanto.cli.client.sdk_client import SdkClient as MemantoClient
    HAS_MEMANTO = True
except ImportError:
    HAS_MEMANTO = False


# ════════════════════════════════════════════════════════════════
# State
# ════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_input: str
    agent_response: str
    memory_context: str
    user_id: str
    should_remember: bool


# ════════════════════════════════════════════════════════════════
# Memanto Memory Backend
# ════════════════════════════════════════════════════════════════

class MemantoBackend:
    """Wraps the Memanto SDK for use in LangGraph nodes.

    Handles agent creation, session activation, and the three
    memory primitives: remember, recall, answer.
    """

    def __init__(self, api_key: str):
        self.client = MemantoClient(api_key=api_key)
        self.agent_ids: Dict[str, str] = {}

    def ensure_agent(self, user_id: str) -> str:
        """Get or create a Memanto agent for this user."""
        if user_id in self.agent_ids:
            return self.agent_ids[user_id]

        # Check if agent already exists
        try:
            agent = self.client.get_agent(user_id)
            agent_id = agent["agent_id"]
        except Exception:
            agent = self.client.create_agent(agent_id=user_id, pattern="support")
            agent_id = agent["agent_id"]

        # Activate session so memory ops work
        self.client.activate_agent(agent_id)
        self.agent_ids[user_id] = agent_id
        return agent_id

    def recall(self, user_id: str, query: str) -> str:
        """Retrieve relevant memories from past sessions."""
        agent_id = self.ensure_agent(user_id)
        try:
            results = self.client.recall(agent_id=agent_id, query=query, limit=10)
            if isinstance(results, list) and results:
                lines = ["Past memories found:"]
                for r in results[:8]:
                    c = r.get("content", r.get("text", ""))
                    t = r.get("title", "")
                    if c:
                        lines.append(f"  [{t}] {c}")
                return "\n".join(lines)
            return "No previous memories found."
        except Exception as e:
            return f"[Memory recall unavailable: {e}]"

    def remember(self, user_id: str, content: str, memory_type: str = "fact") -> bool:
        """Store a memory about this user."""
        agent_id = self.ensure_agent(user_id)
        try:
            title = content[:97] + "..." if len(content) > 100 else content
            self.client.remember(
                agent_id=agent_id,
                memory_type=memory_type,
                title=title,
                content=content[:500],
                confidence=0.9,
                tags=["customer_support", memory_type],
            )
            return True
        except Exception:
            return False

    def answer(self, user_id: str, question: str) -> str:
        """Get an AI-grounded answer directly from memory."""
        agent_id = self.ensure_agent(user_id)
        try:
            result = self.client.answer(agent_id=agent_id, question=question)
            if isinstance(result, dict):
                return result.get("answer", "")
            return str(result)
        except Exception as e:
            return ""


# ════════════════════════════════════════════════════════════════
# Demo Memory Backend (no API key required)
# ════════════════════════════════════════════════════════════════

class DemoMemoryBackend:
    """Simulates Memanto memory for demo purposes.

    Stores memories in a local dictionary. Shows the same
    integration pattern without requiring a Memanto API key.
    """

    def __init__(self):
        self.store: Dict[str, List[Dict]] = {}

    def _get_memories(self, user_id: str) -> list:
        if user_id not in self.store:
            self.store[user_id] = []
        return self.store[user_id]

    def ensure_agent(self, user_id: str) -> str:
        return user_id  # No-op in demo mode

    def recall(self, user_id: str, query: str) -> str:
        memories = self._get_memories(user_id)
        if not memories:
            return "No previous memories found."

        # Simple keyword matching for demo
        q_words = query.lower().split()
        relevant = []
        for m in memories:
            content_lower = m["content"].lower()
            if any(w in content_lower for w in q_words):
                relevant.append(m)

        if not relevant:
            relevant = memories[-3:]  # Fall back to most recent

        lines = ["Relevant information from past interactions:"]
        for m in relevant:
            lines.append(f"  - {m['content']}")
        return "\n".join(lines)

    def remember(self, user_id: str, content: str, memory_type: str = "fact") -> bool:
        self._get_memories(user_id).append({
            "content": content,
            "type": memory_type,
        })
        return True

    def answer(self, user_id: str, question: str) -> str:
        memories = self._get_memories(user_id)
        if not memories:
            return "I don't have any memories about this yet."

        # Simple answer: return the most relevant memory
        q_words = question.lower().split()
        best = None
        best_score = 0
        for m in memories:
            score = sum(1 for w in q_words if w in m["content"].lower())
            if score > best_score:
                best_score = score
                best = m

        if best and best_score > 0:
            return f"Based on our past conversations: {best['content']}"
        return f"I recall that you mentioned: {memories[-1]['content']}"


# ════════════════════════════════════════════════════════════════
# LangGraph Nodes
# ════════════════════════════════════════════════════════════════

def create_graph_nodes(memory):
    """Create LangGraph nodes bound to the memory backend."""

    def recall_node(state: AgentState) -> dict:
        """Node 1: Retrieve memories from past sessions (cross-session recall)."""
        ctx = memory.recall(state["user_id"], state["user_input"])
        return {"memory_context": ctx}

    def respond_node(state: AgentState) -> dict:
        """Node 2: Generate response using memory context.

        If the user is asking about something we should know,
        directly answer from memory using the answer() primitive.
        """
        user_input = state["user_input"]
        memory_context = state["memory_context"]
        user_id = state["user_id"]

        recall_keywords = [
            "remember", "what did i", "what is my", "what's my",
            "do you know", "do you remember", "what have i",
        ]
        is_recall = any(kw in user_input.lower() for kw in recall_keywords)

        if is_recall:
            # Use the answer() primitive for direct Q&A from memory
            answer = memory.answer(user_id, user_input)
            if answer:
                response = f"I remember! {answer}\n\nIs there anything else I can help with?"
            else:
                response = "I don't have any memories about that yet. Tell me more!"
        elif "No previous memories" in memory_context:
            response = (
                "Thanks for reaching out! I'll remember this for our future conversations.\n\n"
                "Is there anything specific you'd like help with?"
            )
        else:
            response_parts = ["I remember our past conversations!", "", memory_context]
            response_parts.append("")
            response_parts.append("Is there anything else I can help you with?")
            response = "\n".join(response_parts)

        should_remember = not is_recall
        return {"agent_response": response, "should_remember": should_remember}

    def store_node(state: AgentState) -> dict:
        """Node 3: Persist information to Memanto for future sessions."""
        if not state["should_remember"]:
            return {}

        user_input = state["user_input"]

        # Classify memory type
        pref_kw = ["prefer", "like", "love", "hate", "allergic", "diet", "vegetarian"]
        info_kw = ["my name", "i am", "my email", "my address", "i live", "my account"]
        goal_kw = ["i want", "i need", "i'm trying", "my goal", "i plan"]

        if any(k in user_input.lower() for k in pref_kw):
            mtype = "preference"
        elif any(k in user_input.lower() for k in info_kw):
            mtype = "instruction"
        elif any(k in user_input.lower() for k in goal_kw):
            mtype = "goal"
        else:
            mtype = "fact"

        memory.remember(state["user_id"], user_input, mtype)
        print(f"  [✓ Stored '{mtype}' memory]")
        return {}

    return recall_node, respond_node, store_node


# ════════════════════════════════════════════════════════════════
# Build Graph
# ════════════════════════════════════════════════════════════════

def build_agent(memory) -> object:
    """Build and compile the LangGraph with shared memory backend."""
    recall_node, respond_node, store_node = create_graph_nodes(memory)

    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_node)
    builder.add_node("respond", respond_node)
    builder.add_node("store", store_node)

    builder.set_entry_point("recall")
    builder.add_edge("recall", "respond")
    builder.add_conditional_edges(
        "respond",
        lambda s: "store" if s["should_remember"] else END,
        {"store": "store", END: END},
    )
    builder.add_edge("store", END)

    app = builder.compile(checkpointer=MemorySaver())
    return app


def run_session(app, user_id: str, user_input: str, thread_id: str = "default") -> str:
    """Run a single LangGraph session (simulates a fresh conversation)."""
    initial = AgentState(
        user_input=user_input,
        agent_response="",
        memory_context="",
        user_id=user_id,
        should_remember=False,
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial, config)
    return result["agent_response"]


def classify_intent(text: str) -> str:
    """Classify the user's input into a memory-relevant category."""
    info_kw = ["my name", "i am", "my email", "my address", "i live", "i have a"]
    pref_kw = ["prefer", "like", "love", "hate", "allergic", "diet", "vegetarian"]
    greet_kw = ["hi", "hello", "hey", "good morning", "good evening"]

    if any(k in text.lower() for k in info_kw):
        return "info"
    if any(k in text.lower() for k in pref_kw):
        return "preference"
    if any(k in text.lower() for k in greet_kw):
        return "greeting"
    return "general"


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Memanto + LangGraph: Cross-Session Memory Demo")
    print("  Each 'session' is a **separate** LangGraph invocation.")
    print("  Memanto remembers what LangGraph forgets.")
    print("=" * 72)

    # Initialize backend
    api_key = os.getenv("MEMANTO_API_KEY", "")
    if api_key and HAS_MEMANTO:
        print(f"\n  Using Memanto backend (live mode)")
        memory = MemantoBackend(api_key)
    else:
        if not HAS_MEMANTO:
            print(f"\n  Memanto SDK not installed. Using demo mode.")
        elif not api_key:
            print(f"\n  No MEMANTO_API_KEY set. Using demo mode.")
        print(f"  For live mode: pip install memanto && export MEMANTO_API_KEY=your_key")
        print()
        memory = DemoMemoryBackend()

    app = build_agent(memory)
    user_id = "demo_user_001"

    # ── Session 1 ──
    print("\n" + "─" * 72)
    print("  SESSION 1 — User shares information")
    print("  (LangGraph state discarded after this block)")
    print("─" * 72)

    s1_messages = [
        "Hi, my name is Alex and I have a premium subscription",
        "I prefer vegetarian meals and I'm allergic to dairy",
        "My shipping address is 123 Main St, Portland, OR 97201",
        "I'm planning to order some specialty cheeses next week",
    ]

    for msg in s1_messages:
        intent = classify_intent(msg)
        print(f"\n  User ({intent}): {msg}")
        response = run_session(app, user_id, msg, thread_id="session_1")
        print(f"  Agent: {response}")

    print("\n  ── Session 1 ends. LangGraph state DISCARDED. ──")

    # ── Session 2 ──
    print("\n" + "─" * 72)
    print("  SESSION 2 — User returns (new LangGraph invocation)")
    print("  LangGraph state is EMPTY. Memanto recalls the past.")
    print("─" * 72)

    s2_messages = [
        "Do you remember me? What's my name?",
        "What's my shipping address?",
        "What allergies do I have?",
        "What did I say about food preferences?",
    ]

    for msg in s2_messages:
        print(f"\n  User: {msg}")
        response = run_session(app, user_id, msg, thread_id="session_2")
        print(f"  Agent: {response}")

    # ── Summary ──
    print()
    print("=" * 72)
    print("  ✓ CROSS-SESSION RECALL DEMONSTRATED")
    print("=" * 72)
    print()
    print("  Session 1: User shared personal info and preferences.")
    print("  Session 2: Agent recalled that info across sessions.")
    print()
    print("  LangGraph provided the workflow state machine.")
    print("  Memanto provided the persistent memory layer.")
    print()
    print("  Three Memanto primitives used:")
    print("    - remember() → persist facts, preferences, goals")
    print("    - recall()   → retrieve relevant past information")
    print("    - answer()   → direct Q&A from stored memory")
    print()
    print("  To run with real Memanto memory:")
    print("    pip install memanto")
    print('    export MEMANTO_API_KEY="your-key"')
    print("    python customer_support.py --live")
    print()


if __name__ == "__main__":
    main()
