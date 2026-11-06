"""定义模型调用边界，并组装模型可见的对话消息。"""

import json
from typing import Any, Mapping

from ops_agent_runtime.prompts import SYSTEM_PROMPT


class ModelConfigurationError(ValueError):
    """表示模型供应商配置缺失或不合法。"""


class ModelRequestError(RuntimeError):
    """表示模型网络请求或供应商 API 调用失败。"""


class ModelOutputError(RuntimeError):
    """表示模型响应无法解析为动作对象。"""


def build_model_messages(
    state: Mapping[str, Any],
    max_investigation_count: int,
) -> list[dict[str, str]]:
    """组装模型可见消息，并排除 traces、最终结论等内部状态。"""
    state_snapshot = {
        "type": "state_snapshot",
        "investigation_count": state["investigation_count"],
        "max_investigation_count": max_investigation_count,
        "evidence": state["evidence"],
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["user_question"]},
        *state["messages"],
        {
            "role": "user",
            "content": json.dumps(
                state_snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
