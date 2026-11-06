from copy import deepcopy

from ops_agent_runtime.agent import build_agent_graph


class MetricToolModel:
    """先查询真实指标，再引用其中一个真实指标点提交结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, messages):
        self.contexts.append(deepcopy(messages))
        if len(self.contexts) == 1:
            return {
                "action": "call_tool",
                "tool": "query_metrics",
                "arguments": {
                    "metric": "error_rate",
                    "scope": "global",
                    "start": "2026-11-01T12:00",
                    "end": "2026-11-01T12:10",
                },
                "reason": "查询全局错误率确认故障影响",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "支付配置错误导致全局错误率升高",
            "evidence": [
                {
                    "type": "metric",
                    "metric": "error_rate",
                    "scope": "global",
                    "time": "2026-11-01T12:05",
                }
            ],
            "confidence": "high",
            "unknowns": [],
        }


def test_metric_tool_result_generates_evidence_and_reaches_finalize():
    """指标工具应进入 Graph，并把真实指标点作为最终结论证据。"""
    model = MetricToolModel()
    graph = build_agent_graph(model)

    result = graph.invoke(
        {
            "user_question": "下单一直失败，帮我排查原因",
            "messages": [],
            "current_action": None,
            "investigation_count": 0,
            "called_signatures": [],
            "evidence": [],
            "traces": [],
            "investigation_status": "investigating",
            "final_conclusion": None,
        }
    )

    assert len(model.contexts) == 2
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert result["evidence"] == [
        {
            "type": "metric",
            "metric": "error_rate",
            "scope": "global",
            "time": "2026-11-01T12:00",
        },
        {
            "type": "metric",
            "metric": "error_rate",
            "scope": "global",
            "time": "2026-11-01T12:05",
        },
        {
            "type": "metric",
            "metric": "error_rate",
            "scope": "global",
            "time": "2026-11-01T12:10",
        },
    ]
    assert any(
        event["event"] == "tool_result"
        and event["tool"] == "query_metrics"
        and len(event["result"]["points"]) == 3
        for event in result["traces"]
    )


class DeployToolModel:
    """先查询真实发布记录，再引用默认场景中的支付发布提交结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, messages):
        self.contexts.append(deepcopy(messages))
        if len(self.contexts) == 1:
            return {
                "action": "call_tool",
                "tool": "query_deploys",
                "arguments": {
                    "service": "payment",
                    "start": "2026-11-01T12:00",
                    "end": "2026-11-01T12:00",
                },
                "reason": "查询支付服务在故障开始前的发布记录",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "支付服务发布修改了购物车连接端口",
            "evidence": [{"type": "deploy", "id": "dep_d1003"}],
            "confidence": "high",
            "unknowns": [],
        }


def test_deploy_tool_result_generates_evidence_and_reaches_finalize():
    """发布工具应进入 Graph，并把真实发布 ID 作为最终结论证据。"""
    model = DeployToolModel()
    graph = build_agent_graph(model)

    result = graph.invoke(
        {
            "user_question": "下单一直失败，帮我排查原因",
            "messages": [],
            "current_action": None,
            "investigation_count": 0,
            "called_signatures": [],
            "evidence": [],
            "traces": [],
            "investigation_status": "investigating",
            "final_conclusion": None,
        }
    )

    assert len(model.contexts) == 2
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert result["evidence"] == [{"type": "deploy", "id": "dep_d1003"}]
    assert any(
        event["event"] == "tool_result"
        and event["tool"] == "query_deploys"
        and event["result"]["total"] == 1
        and event["result"]["deploys"][0]["id"] == "dep_d1003"
        for event in result["traces"]
    )
