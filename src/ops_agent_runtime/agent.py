from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ops_agent_runtime.tools import query_logs


class AgentState(TypedDict):
    """一次故障调查在主循环中的最小状态。"""

    user_question: str                               #用户问题
    messages: Annotated[list[dict], add]             #消息列表
    current_action: dict | None                      #当前操作
    investigation_count: int                         #调查次数
    called_signatures: Annotated[list[str], add]     #已用工具签名列表
    evidence: Annotated[list[dict], add]             #证据列表
    traces: Annotated[list[dict], add]               #追踪记录
    investigation_status: str                        #调查状态
    final_conclusion: dict | None                    #最终结论


def build_agent_graph(model):
    """构建“模型决策 -> 工具执行 -> 模型决策”的最小 LangGraph。"""

    def agent_node(state: AgentState):
        action = model(state)

        return {
            "current_action": action,
            "traces": [
                {
                    "event": "model_decision",
                    "action": action.get("action"),
                    "tool": action.get("tool"),
                    "arguments": action.get("arguments"),
                    "reason": action.get("reason"),
                }
            ],
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

        evidence = [
            *(
                {"type": "log", "id": log["id"]}
                for log in result.get("logs", [])
            ),
        ]

        return {
            "messages": [message],
            "current_action": None,
            "investigation_count": state["investigation_count"] + 1,
            "evidence": evidence,
            "traces": [
                {
                    "event": "tool_result",
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            ],
        }

    def finalize_node(state: AgentState):
        action = state["current_action"]

        findings = {
            "root_cause": action.get("root_cause"),
            "evidence": action.get("evidence", []),
            "confidence": action.get("confidence"),
            "unknowns": action.get("unknowns", []),
        }

        return {
            "current_action": None,
            "investigation_status": "completed",
            "final_conclusion": findings,
            "traces": [
                {
                    "event": "final_conclusion",
                    "conclusion": findings,
                }
            ],
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
