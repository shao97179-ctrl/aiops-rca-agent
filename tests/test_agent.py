from copy import deepcopy

from ops_agent_runtime.agent import build_agent_graph


class FakeModel:
    """按两轮脚本模拟“先查日志、再提交结论”的模型。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        # 保存状态快照，用于验证第二轮模型确实拿到了工具结果。
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


class InvalidActionModel:
    """第一轮返回非法动作，第二轮修正为低置信度结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        # 保留每次上下文，验证校验错误确实回灌给了下一轮模型。
        self.contexts.append(deepcopy(state))
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
    assert "不支持的动作" in repr(model.contexts[1]["messages"])
    assert result["investigation_count"] == 1
    assert result["investigation_status"] == "completed"
    assert any(
        event["event"] == "validation_error"
        and "不支持的动作" in event["error"]
        for event in result["traces"]
    )


class DuplicateCallModel:
    """连续提出相同工具调用，再提交低置信度结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        if len(self.contexts) <= 2:
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
            "root_cause": "尚未确定根因",
            "evidence": [],
            "confidence": "low",
            "unknowns": ["重复查询被拦截，现有证据不足"],
        }


def test_duplicate_tool_call_is_blocked_and_counted():
    """相同工具和参数只执行一次，第二次尝试计数并回灌错误。"""
    model = DuplicateCallModel()
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

    assert len(model.contexts) == 3
    assert "重复工具调用已拦截" in repr(model.contexts[2]["messages"])
    assert result["investigation_count"] == 2
    assert len(result["called_signatures"]) == 1
    assert len(result["evidence"]) == 4
    assert sum(event["event"] == "tool_result" for event in result["traces"]) == 1
    assert any(
        event["event"] == "duplicate_call_blocked"
        and event["tool"] == "query_logs"
        for event in result["traces"]
    )


class NeverConcludingModel:
    """持续发起不同参数的查询，始终不提交结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        minute = len(self.contexts) - 1
        return {
            "action": "call_tool",
            "tool": "query_logs",
            "arguments": {
                "service": "payment",
                "level": "ERROR",
                "start": f"2026-11-01T12:{minute:02d}",
                "end": f"2026-11-01T12:{minute:02d}",
            },
            "reason": "继续查询不同时间的支付服务错误日志",
        }


def test_investigation_stops_after_eight_attempts_without_conclusion():
    """第 8 次尝试后继续调用工具时，调查应标记未完成并终止。"""
    model = NeverConcludingModel()
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

    assert len(model.contexts) == 9
    assert result["investigation_count"] == 8
    assert result["investigation_status"] == "incomplete"
    assert result["final_conclusion"] is None
    assert sum(event["event"] == "tool_result" for event in result["traces"]) == 8
    assert any(
        event["event"] == "investigation_limit_reached"
        and event["max_investigation_count"] == 8
        for event in result["traces"]
    )


class ConcludeAtLimitModel:
    """执行第 8 次查询后，利用已有结果提交结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        if len(self.contexts) <= 8:
            minute = len(self.contexts) - 1
            return {
                "action": "call_tool",
                "tool": "query_logs",
                "arguments": {
                    "service": "payment",
                    "level": "ERROR",
                    "start": f"2026-11-01T12:{minute:02d}",
                    "end": f"2026-11-01T12:{minute:02d}",
                },
                "reason": "继续查询不同时间的支付服务错误日志",
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


def test_conclusion_can_be_submitted_after_eighth_attempt():
    """第 8 次查询结果回灌后，仍允许模型提交最终结论。"""
    model = ConcludeAtLimitModel()
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

    assert len(model.contexts) == 9
    assert result["investigation_count"] == 8
    assert result["investigation_status"] == "completed"
    assert result["final_conclusion"]["root_cause"] == "支付服务连接购物车的端口配置错误"
    assert not any(
        event["event"] == "investigation_limit_reached"
        for event in result["traces"]
    )


class InvalidAtLimitModel:
    """已尝试 7 次时返回非法动作，不应获得第 9 次机会。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        if len(self.contexts) == 1:
            return {
                "action": "delete_database",
                "reason": "尝试执行不存在的危险动作",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "不应执行到这里",
            "evidence": [],
            "confidence": "low",
            "unknowns": ["非法动作已经达到尝试上限"],
        }


def test_invalid_action_at_limit_ends_without_another_retry():
    """第 8 次尝试为非法动作时，应直接标记未完成并终止。"""
    model = InvalidAtLimitModel()
    graph = build_agent_graph(model)

    result = graph.invoke(
        {
            "user_question": "系统出现异常，请排查",
            "messages": [],
            "current_action": None,
            "investigation_count": 7,
            "called_signatures": [],
            "evidence": [],
            "traces": [],
            "investigation_status": "investigating",
            "final_conclusion": None,
        }
    )

    assert len(model.contexts) == 1
    assert result["investigation_count"] == 8
    assert result["investigation_status"] == "incomplete"
    assert result["final_conclusion"] is None
    assert any(
        event["event"] == "investigation_limit_reached"
        for event in result["traces"]
    )
