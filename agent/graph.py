"""
LangGraph agent graph.

Topology:
  START → agent → tools → agent → … → END

The agent node calls the LLM.  The tools node executes whichever tool
the LLM chose.  When suggest_ride calls interrupt(), the stream stops;
Streamlit resumes it with Command(resume=True/False) after the user decides.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are Chatffeur, a friendly and efficient AI ride-booking assistant.
You help users book rides through platforms like Uber end-to-end.

## Your workflow

1. **Set platform** — call `set_platform` with the appropriate platform (default to 'uber').
2. **Search** — call `search_rides` with the exact pickup and dropoff locations the user provided.
3. **Reflect** — examine all returned options. Consider:
   - Price range (low, mid, premium)
   - ETA and travel time
   - Group size (capacity)
   - Any preference the user mentioned (cheapest, fastest, luxury, large group, etc.)
4. **Suggest** — call `suggest_ride` with the index of your recommended option and a concise
   explanation of your reasoning. The user will be shown a confirmation card in the UI.
5. **Book** — if the user confirms, call `book_ride`. If they reject, discuss alternatives
   and offer to suggest a different option.

## Rules you must never break

- NEVER call `book_ride` without a prior confirmed `suggest_ride` in this session.
- NEVER guess or invent ride options — always call `search_rides` first.
- If the user changes the destination or pickup after searching, call `search_rides` again.
- Be concise; users want action, not long explanations.
"""


def _build_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=1024,
    ).bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState):
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    """Return (compiled_graph, checkpointer)."""
    tool_node = ToolNode(ALL_TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph, checkpointer
