from copy import deepcopy

import pytest

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


class ConclusionOnlyModel:
    """不调用工具，直接提交指定置信度和证据的结论。"""

    def __init__(self, confidence, evidence, unknowns):
        self.confidence = confidence
        self.evidence = evidence
        self.unknowns = unknowns
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
        return {
            "action": "submit_conclusion",
            "root_cause": "测试结论",
            "evidence": self.evidence,
            "confidence": self.confidence,
            "unknowns": self.unknowns,
        }


@pytest.mark.parametrize("confidence", ["high", "medium"])
def test_high_or_medium_confidence_requires_evidence(confidence):
    """高、中置信度没有真实证据时，最终结论应判定失败。"""
    model = ConclusionOnlyModel(confidence, [], [])
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

    assert len(model.contexts) == 1
    assert result["investigation_status"] == "failed"
    assert result["final_conclusion"] is None
    assert any(
        event["event"] == "final_validation_error"
        and "高或中置信度结论必须引用真实证据" in event["errors"]
        for event in result["traces"]
    )


def test_low_confidence_requires_unknowns():
    """低置信度结论没有未知项时，最终结论应判定失败。"""
    model = ConclusionOnlyModel("low", [], [])
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

    assert len(model.contexts) == 1
    assert result["investigation_status"] == "failed"
    assert result["final_conclusion"] is None
    assert any(
        event["event"] == "final_validation_error"
        and "低置信度结论必须填写未知项" in event["errors"]
        for event in result["traces"]
    )


class FabricatedEvidenceModel:
    """查询真实日志后，提交一条本次未返回的虚构日志证据。"""

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
                "reason": "先查询真实日志证据",
            }
        return {
            "action": "submit_conclusion",
            "root_cause": "支付服务连接购物车的端口配置错误",
            "evidence": [{"type": "log", "id": "log_9999"}],
            "confidence": "high",
            "unknowns": [],
        }


def test_fabricated_evidence_fails_validation():
    """结论引用本次会话未返回的证据时，整次调查应判定失败。"""
    model = FabricatedEvidenceModel()
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
    assert result["investigation_status"] == "failed"
    assert result["final_conclusion"] is None
    assert any(
        event["event"] == "final_validation_error"
        and "本次调查未返回的证据" in event["errors"][0]
        and "log_9999" in event["errors"][0]
        for event in result["traces"]
    )


class MetricToolModel:
    """先查询真实指标，再引用其中一个真实指标点提交结论。"""

    def __init__(self):
        self.contexts = []

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
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

    def __call__(self, state):
        self.contexts.append(deepcopy(state))
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
