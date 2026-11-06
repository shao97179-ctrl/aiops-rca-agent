import json
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ops_agent_runtime.tools import query_logs


class QueryLogsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str | None = None
    level: Literal["INFO", "WARN", "ERROR"] | None = None
    start: str | None = None
    end: str | None = None


class CallToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"]
    tool: Literal["query_logs"]
    arguments: QueryLogsArguments
    reason: str = Field(min_length=1)


class LogEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["log"]
    id: str = Field(min_length=1)


class DeployEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["deploy"]
    id: str = Field(min_length=1)


class MetricEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["metric"]
    metric: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    time: str = Field(min_length=1)


EvidenceReference = Annotated[
    LogEvidence | DeployEvidence | MetricEvidence,
    Field(discriminator="type"),
]


class SubmitConclusionAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["submit_conclusion"]
    root_cause: str = Field(min_length=1)
    evidence: list[EvidenceReference]
    confidence: Literal["high", "medium", "low"]
    unknowns: list[str]


ModelDecision = Annotated[
    CallToolAction | SubmitConclusionAction,
    Field(discriminator="action"),
]
MODEL_DECISION_ADAPTER = TypeAdapter(ModelDecision)
MAX_INVESTIGATION_ATTEMPTS = 8


def _parse_model_action(raw_action):
    """解析模型原始输出；失败时返回可回灌的错误，而不是抛异常。"""
    if not isinstance(raw_action, dict):
        return None, (
            "模型输出不合法：应为对象，"
            f"实际收到 {type(raw_action).__name__}"
        )

    action_name = raw_action.get("action")
    try:
        return MODEL_DECISION_ADAPTER.validate_python(raw_action), None
    except ValidationError as exc:
        if action_name not in {"call_tool", "submit_conclusion"}:
            return None, f"不支持的动作：{action_name!r}"
        return None, f"动作 {action_name!r} 的模型输出不合法：{exc}"


def _tool_call_signature(tool_name, arguments):
    """生成稳定的工具调用签名，用于识别工具名和参数都相同的调用。"""
    return json.dumps(
        {"tool": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
        parsed_action, error = _parse_model_action(model(state))

        if error:
            return {
                "current_action": None,
                "messages": [{"role": "validation_error", "error": error}],
                "investigation_count": min(
                    state["investigation_count"] + 1,
                    MAX_INVESTIGATION_ATTEMPTS,
                ),
                "traces": [{"event": "validation_error", "error": error}],
            }

        action = parsed_action.model_dump()
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

        if action and action.get("action") == "submit_conclusion":
            return "submit_conclusion"

        if state["investigation_count"] >= MAX_INVESTIGATION_ATTEMPTS:
            return "limit_reached"

        if action is None:
            return "retry"

        if action.get("action") == "call_tool":
            return "call_tool"

        raise ValueError(f"不支持的动作：{action!r}")

    def tool_node(state: AgentState):
        action = state["current_action"]
        tool_name = action.get("tool")
        arguments = action.get("arguments", {})

        if tool_name != "query_logs":
            raise ValueError(f"不支持的工具：{tool_name!r}")

        signature = _tool_call_signature(tool_name, arguments)
        if signature in state["called_signatures"]:
            error = (
                f"重复工具调用已拦截：{tool_name} "
                "已使用相同参数调用过"
            )
            return {
                "messages": [
                    {
                        "role": "tool_call_error",
                        "tool": tool_name,
                        "arguments": arguments,
                        "error": error,
                    }
                ],
                "current_action": None,
                "investigation_count": state["investigation_count"] + 1,
                "traces": [
                    {
                        "event": "duplicate_call_blocked",
                        "tool": tool_name,
                        "arguments": arguments,
                        "signature": signature,
                        "error": error,
                    }
                ],
            }

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
            "called_signatures": [signature],
            "evidence": evidence,
            "traces": [
                {
                    "event": "tool_result",
                    "tool": tool_name,
                    "arguments": arguments,
                    "signature": signature,
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

    def limit_reached_node(state: AgentState):
        return {
            "current_action": None,
            "investigation_status": "incomplete",
            "final_conclusion": None,
            "traces": [
                {
                    "event": "investigation_limit_reached",
                    "investigation_count": state["investigation_count"],
                    "max_investigation_count": MAX_INVESTIGATION_ATTEMPTS,
                }
            ],
        }

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("limit", limit_reached_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "call_tool": "tools",
            "submit_conclusion": "finalize",
            "retry": "agent",
            "limit_reached": "limit",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    graph.add_edge("limit", END)

    return graph.compile()
