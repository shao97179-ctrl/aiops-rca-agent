"""提供工具调用去重和最终结论真实性的确定性校验。"""

import json


def _canonical_json(value) -> str:
    """把简单数据结构转换为字段顺序稳定的紧凑 JSON 字符串。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tool_call_signature(tool_name: str, arguments: dict) -> str:
    """生成稳定的工具调用签名，用于识别工具名和参数都相同的调用。"""
    return _canonical_json({"tool": tool_name, "arguments": arguments})


def evidence_reference_key(evidence: dict) -> str:
    """把单条证据转换为规范字符串，用于判断它是否由本次会话真实返回。"""
    return _canonical_json(evidence)


def validate_conclusion(
    action: dict,
    available_evidence: list[dict],
) -> list[str]:
    """校验最终结论的证据真实性和置信度约束，返回中文错误列表。"""
    submitted_evidence = action.get("evidence", [])
    available_keys = {
        evidence_reference_key(evidence) for evidence in available_evidence
    }
    unknown_evidence = [
        evidence
        for evidence in submitted_evidence
        if evidence_reference_key(evidence) not in available_keys
    ]

    errors = []
    if unknown_evidence:
        errors.append(f"结论引用了本次调查未返回的证据：{unknown_evidence}")

    confidence = action.get("confidence")
    if confidence in {"high", "medium"} and not submitted_evidence:
        errors.append("高或中置信度结论必须引用真实证据")
    if confidence == "low" and not action.get("unknowns"):
        errors.append("低置信度结论必须填写未知项")
    return errors
