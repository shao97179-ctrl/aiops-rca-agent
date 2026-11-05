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


def _parse_model_action(raw_action):
    """解析模型原始输出；失败时返回可回灌的错误，而不是抛异常。"""
    if not isinstance(raw_action, dict):
        return None, (
            "invalid model output: expected an object, "
            f"received {type(raw_action).__name__}"
        )

    action_name = raw_action.get("action")
    try:
        return MODEL_DECISION_ADAPTER.validate_python(raw_action), None
    except ValidationError as exc:
        if action_name not in {"call_tool", "submit_conclusion"}:
            return None, f"unsupported action: {action_name!r}"
        return None, f"invalid model output for action {action_name!r}: {exc}"


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
                "investigation_count": state["investigation_count"] + 1,
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

        if action is None:
            return "retry"

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
            "retry": "agent",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
