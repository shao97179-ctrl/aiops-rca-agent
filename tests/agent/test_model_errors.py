import pytest

from ops_agent_runtime.agent import build_agent_graph
from ops_agent_runtime.model import (
    ModelConfigurationError,
    ModelOutputError,
    ModelRequestError,
)


def _initial_state(investigation_count=0, evidence=None):
    return {
        "user_question": "系统出现异常，请排查",
        "messages": [],
        "current_action": None,
        "investigation_count": investigation_count,
        "called_signatures": [],
        "evidence": [] if evidence is None else evidence,
        "traces": [],
        "investigation_status": "investigating",
        "final_conclusion": None,
    }


class FailingModel:
    """在第一次模型调用时抛出指定异常。"""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ModelConfigurationError("模型配置不合法"),
        ModelRequestError("模型请求失败"),
        ModelOutputError("模型输出不合法"),
    ],
)
def test_expected_model_error_terminates_with_system_error(error):
    """模型配置、请求和输出错误应受控终止，而不是从 Graph 向外抛出。"""
    model = FailingModel(error)
    graph = build_agent_graph(model)

    result = graph.invoke(_initial_state())

    assert model.calls == 1
    assert result["investigation_status"] == "system_error"
    assert result["investigation_count"] == 0
    assert result["final_conclusion"] is None
    assert any(
        event["event"] == "model_error"
        and event["error_type"] == type(error).__name__
        and event["error"] == str(error)
        for event in result["traces"]
    )


def test_model_error_preserves_previous_investigation_state():
    """模型错误不应抹掉此前真实证据和调查次数。"""
    evidence = [{"type": "log", "id": "log_0011"}]
    model = FailingModel(ModelRequestError("模型请求失败"))
    graph = build_agent_graph(model)

    result = graph.invoke(
        _initial_state(investigation_count=2, evidence=evidence)
    )

    assert result["investigation_count"] == 2
    assert result["evidence"] == evidence
    assert result["investigation_status"] == "system_error"
    assert result["final_conclusion"] is None


def test_unexpected_model_exception_is_not_swallowed():
    """未预期异常应继续向外抛出，避免掩盖适配器或程序缺陷。"""
    model = FailingModel(RuntimeError("程序缺陷"))
    graph = build_agent_graph(model)

    with pytest.raises(RuntimeError, match="程序缺陷"):
        graph.invoke(_initial_state())
