from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ops_agent_runtime.tools import query_logs


class AgentState(TypedDict):
    """一次故障调查在主循环中的最小状态。"""

    user_question: str
    messages: list[dict]
    current_action: dict | None
    investigation_count: int
    called_signatures: list[str]
    evidence: list[dict]
    traces: list[dict]
    investigation_status: str
    final_conclusion: dict | None


def build_agent_graph(model):
    """构建“模型决策 -> 工具执行 -> 模型决策”的最小 LangGraph。"""

    def agent_node(state: AgentState):
        action = model(state)

        trace = [
            *state["traces"],
            {
                "event": "model_decision",
                "action": action.get("action"),
                "tool": action.get("tool"),
                "arguments": action.get("arguments"),
                "reason": action.get("reason"),
            },
        ]

        return {
            "current_action": action,
            "traces": trace,
        }

    def route_after_agent(state: AgentState):
        action = state["current_action"]

        if action and action.get("action") == "call_tool":
            return "call_tool"

        if action and action.get("action") == "submit_conclusion":
            return "submit_conclusion"

        raise ValueError(f"unsupported action: {action!r}")

    def tool_node(state: AgentState):
        action = state["current_action"]
        tool_name = action.get("tool")
        arguments = action.get("arguments", {})

        if tool_name != "query_logs":
            raise ValueError(f"unsupported tool: {tool_name!r}")

        result = query_logs(
            service=arguments.get("service"),
            level=arguments.get("level"),
            start=arguments.get("start"),
            end=arguments.get("end"),
        )

        message = {
            "role": "tool",
            "tool": tool_name,
            "arguments": arguments,
            "reason": action.get("reason"),
            "result": result,
        }

        trace = [
            *state["traces"],
            {
                "event": "tool_result",
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            },
        ]

        evidence = [
            *state["evidence"],
            *(
                {"type": "log", "id": log["id"]}
                for log in result.get("logs", [])
            ),
        ]

        return {
            "messages": [*state["messages"], message],
            "current_action": None,
            "investigation_count": state["investigation_count"] + 1,
            "evidence": evidence,
            "traces": trace,
        }

    def finalize_node(state: AgentState):
        action = state["current_action"]

        findings = {
            "root_cause": action.get("root_cause"),
            "evidence": action.get("evidence", []),
            "confidence": action.get("confidence"),
            "unknowns": action.get("unknowns", []),
        }

        trace = [
            *state["traces"],
            {
                "event": "final_conclusion",
                "conclusion": findings,
            },
        ]

        return {
            "current_action": None,
            "investigation_status": "completed",
            "final_conclusion": findings,
            "traces": trace,
        }

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "call_tool": "tools",
            "submit_conclusion": "finalize",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
