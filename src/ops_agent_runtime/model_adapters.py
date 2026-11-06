"""提供 GLM 和 DeepSeek 共用的 OpenAI 兼容模型适配器。"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from ops_agent_runtime.model import (
    ModelConfigurationError,
    ModelOutputError,
    ModelRequestError,
)



PROVIDER_BASE_URLS = {
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com",
}
JSON_CODE_FENCE = re.compile(
    r"\A```(?:json)?\s*(.*?)\s*```\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """保存调用 OpenAI 兼容服务所需的非默认配置。"""

    provider: str
    api_key: str = field(repr=False)
    model: str
    base_url: str
    timeout_seconds: float = 30.0


def load_provider_config(
    provider: str,
    environ: Mapping[str, str] | None = None,
    env_file: str | os.PathLike[str] | None = None,
) -> OpenAICompatibleConfig:
    """从供应商特定的环境变量读取 API key、模型名称和 base URL。"""
    provider_name = provider.strip().lower()
    default_base_url = PROVIDER_BASE_URLS.get(provider_name)
    if default_base_url is None:
        supported = "、".join(sorted(PROVIDER_BASE_URLS))
        raise ModelConfigurationError(
            f"不支持的模型供应商：{provider!r}；当前支持：{supported}"
        )

    if environ is None:
        selected_env_file = env_file or find_dotenv(usecwd=True)
        if selected_env_file:
            load_dotenv(selected_env_file, override=False)
        env = os.environ
    else:
        env = environ
    prefix = provider_name.upper()
    api_key_name = f"{prefix}_API_KEY"
    model_name = f"{prefix}_MODEL"
    base_url_name = f"{prefix}_BASE_URL"

    api_key = env.get(api_key_name, "").strip()
    model = env.get(model_name, "").strip()
    missing = [
        name
        for name, value in ((api_key_name, api_key), (model_name, model))
        if not value
    ]
    if missing:
        raise ModelConfigurationError(
            f"缺少模型配置环境变量：{'、'.join(missing)}"
        )

    base_url = env.get(base_url_name, "").strip() or default_base_url
    return OpenAICompatibleConfig(
        provider=provider_name,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


class OpenAICompatibleModel:
    """通过 OpenAI SDK 调用 GLM 或 DeepSeek，并返回 JSON 动作字典。"""

    def __init__(self, config: OpenAICompatibleConfig, client=None):
        self.config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def __call__(self, messages: list[dict[str, str]]) -> dict:
        """发送消息并解析模型返回的单个 JSON 动作对象。"""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
            )
        except Exception as exc:
            raise ModelRequestError(
                f"{self.config.provider} 模型请求失败"
            ) from exc

        content = _extract_response_content(response)
        return _parse_action_content(content)


def build_model_from_env(
    provider: str,
    environ: Mapping[str, str] | None = None,
    client=None,
    env_file: str | os.PathLike[str] | None = None,
) -> OpenAICompatibleModel:
    """根据供应商名称和环境变量构造可直接传给 Agent 图的模型。"""
    return OpenAICompatibleModel(
        load_provider_config(provider, environ, env_file),
        client=client,
    )


def _extract_response_content(response) -> str:
    """从聊天补全响应中提取非空文本内容。"""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelOutputError("模型响应缺少 choices[0].message.content") from exc

    if not isinstance(content, str) or not content.strip():
        raise ModelOutputError("模型响应内容为空或不是文本")
    return content


def _parse_action_content(content: str) -> dict:
    """解析普通 JSON 或 Markdown JSON 代码块，并确保结果是对象。"""
    text = content.strip()
    code_fence = JSON_CODE_FENCE.fullmatch(text)
    if code_fence:
        text = code_fence.group(1).strip()

    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelOutputError("模型响应不是合法 JSON") from exc

    if not isinstance(action, dict):
        raise ModelOutputError("模型响应 JSON 必须是对象")
    return action
