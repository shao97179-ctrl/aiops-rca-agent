from copy import deepcopy

from ops_agent_runtime.agent import build_agent_graph


class FakeModel:
    """按两轮脚本模拟“先查日志、再提交结论”的模型。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, messages):
        # 保存状态快照，用于验证第二轮模型确实拿到了工具结果。
        self.contexts.append(deepcopy(messages))
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
    """验证日志查询结果会回灌给模型，然后模型才能提交结论。"""
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
    assert [message["role"] for message in model.contexts[1]] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert "log_0011" in repr(model.contexts[1])
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert result["final_conclusion"]["root_cause"] == "支付服务连接购物车的端口配置错误"
    assert any(
        event["event"] == "tool_result"
        and event["tool"] == "query_logs"
        and event["result"]["total"] == 4
        for event in result["traces"]
    )


class InvalidActionModel:
    """第一轮返回非法动作，第二轮修正为低置信度结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, messages):
        # 保留每次上下文，验证校验错误确实回灌给了下一轮模型。
        self.contexts.append(deepcopy(messages))
        if len(self.contexts) == 1:
            return {
                "action": "delete_database",
                "reason": "尝试执行不存在的危险动作",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "尚未确定根因",
            "evidence": [],
            "confidence": "low",
            "unknowns": ["没有获得可用的调查证据"],
        }


def test_invalid_action_is_returned_to_model_instead_of_crashing():
    """验证非法动作进入受控重试路径，而不是让整个图崩溃。"""
    model = InvalidActionModel()
    graph = build_agent_graph(model)

    result = graph.invoke(
        {
            "user_question": "系统出现异常，请排查",
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
    assert "不支持的动作" in repr(model.contexts[1])
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert any(
        event["event"] == "validation_error"
        and "不支持的动作" in event["error"]
        for event in result["traces"]
    )
