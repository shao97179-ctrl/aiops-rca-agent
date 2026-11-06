from copy import deepcopy

import pytest

from ops_agent_runtime.agent import build_agent_graph


class ConclusionOnlyModel:
    """不调用工具，直接提交指定置信度和证据的结论。"""

    def __init__(self, confidence, evidence, unknowns):
        self.confidence = confidence
        self.evidence = evidence
        self.unknowns = unknowns
        self.contexts = []

    def __call__(self, messages):
        self.contexts.append(deepcopy(messages))
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

    def __call__(self, messages):
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
