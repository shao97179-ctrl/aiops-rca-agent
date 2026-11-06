from ops_agent_runtime.model import build_model_messages
from ops_agent_runtime.prompts import SYSTEM_PROMPT


def test_model_messages_include_history_and_state_but_not_internal_traces():
    """模型消息应包含公开上下文，并排除 traces 和最终结论。"""
    state = {
        "user_question": "系统出现异常，请排查",
        "messages": [
            {
                "role": "assistant",
                "content": '{"action":"call_tool","tool":"query_logs"}',
            },
            {
                "role": "user",
                "content": '{"type":"tool_result","result":{"logs":[]}}',
            },
        ],
        "investigation_count": 1,
        "evidence": [{"type": "log", "id": "log_0011"}],
        "traces": [{"event": "internal_trace_marker"}],
        "final_conclusion": {"root_cause": "internal_conclusion_marker"},
    }

    messages = build_model_messages(state, max_investigation_count=8)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "系统出现异常，请排查"}
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert all(set(message) == {"role", "content"} for message in messages)
    assert "log_0011" in messages[-1]["content"]
    assert "internal_trace_marker" not in repr(messages)
    assert "internal_conclusion_marker" not in repr(messages)


def test_system_prompt_contains_rules_without_scenario_answers():
    """系统提示词应声明工具和证据规则，同时不泄漏场景专属答案。"""
    required_rules = [
        "每轮只生成一个动作",
        "query_logs",
        "query_metrics",
        "query_deploys",
        "只能引用本次会话中真实返回的证据",
        "高或中置信度必须引用真实证据",
        "低置信度必须填写非空未知项",
        "只能执行查询操作",
    ]
    scenario_answers = [
        "payment",
        "cart",
        "8081",
        "dep_d1003",
        "log_0011",
        "error_rate",
    ]

    for rule in required_rules:
        assert rule in SYSTEM_PROMPT
    for answer in scenario_answers:
        assert answer not in SYSTEM_PROMPT
