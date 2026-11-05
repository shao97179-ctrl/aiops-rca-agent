from copy import deepcopy

from ops_agent_runtime.agent import build_agent_graph


class FakeModel:
    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        if len(self.contexts) == 1:
            return {
                "action": "call_tool",
                "tool": "query_logs",
                "arguments": {
                    "service": "payment",
                    "level": "ERROR",
                    "start": "2026-11-01T12:05",
                    "end": "2026-11-01T12:05",
                },
                "reason": "支付服务错误日志可以确认故障表现",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "支付服务连接购物车的端口配置错误",
            "evidence": [
                {"type": "log", "id": "log_0011"},
                {"type": "log", "id": "log_0013"},
            ],
            "confidence": "high",
            "unknowns": [],
        }


def test_log_tool_result_returns_to_model_before_submission():
    model = FakeModel()
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
    assert "log_0011" in repr(model.contexts[1]["messages"])
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert result["final_conclusion"]["root_cause"] == "支付服务连接购物车的端口配置错误"
    assert any(
        event["event"] == "tool_result"
        and event["tool"] == "query_logs"
        and event["result"]["total"] == 4
        for event in result["traces"]
    )
