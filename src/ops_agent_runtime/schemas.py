"""定义模型动作、工具参数和证据引用的 Pydantic 机器协议。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class QueryLogsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str | None = None
    level: Literal["INFO", "WARN", "ERROR"] | None = None
    start: str | None = None
    end: str | None = None


class QueryMetricsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    start: str | None = None
    end: str | None = None


class QueryDeploysArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str | None = None
    start: str | None = None
    end: str | None = None


class QueryLogsAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"]
    tool: Literal["query_logs"]
    arguments: QueryLogsArguments
    reason: str = Field(min_length=1)


class QueryMetricsAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"]
    tool: Literal["query_metrics"]
    arguments: QueryMetricsArguments
    reason: str = Field(min_length=1)


class QueryDeploysAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"]
    tool: Literal["query_deploys"]
    arguments: QueryDeploysArguments
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


CallToolAction = QueryLogsAction | QueryMetricsAction | QueryDeploysAction
ModelDecision = CallToolAction | SubmitConclusionAction

CALL_TOOL_ACTION_ADAPTERS = {
    "query_logs": TypeAdapter(QueryLogsAction),
    "query_metrics": TypeAdapter(QueryMetricsAction),
    "query_deploys": TypeAdapter(QueryDeploysAction),
}
SUBMIT_CONCLUSION_ADAPTER = TypeAdapter(SubmitConclusionAction)
