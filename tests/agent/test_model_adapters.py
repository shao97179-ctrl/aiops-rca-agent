from types import SimpleNamespace

import pytest

from ops_agent_runtime.model import (
    ModelConfigurationError,
    ModelOutputError,
    ModelRequestError,
)
from ops_agent_runtime.model_adapters import (
    OpenAICompatibleConfig,
    OpenAICompatibleModel,
    build_model_from_env,
    load_provider_config,
)


class FakeCompletions:
    """记录请求并返回预置聊天补全响应。"""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeOpenAIClient:
    """模拟 OpenAI SDK 的最小客户端结构。"""

    def __init__(self, content=None, error=None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(content=content, error=error),
        )


def _config(provider="glm"):
    return OpenAICompatibleConfig(
        provider=provider,
        api_key="test-secret",
        model=f"{provider}-test-model",
        base_url=f"https://{provider}.example.invalid/v4",
    )


@pytest.mark.parametrize(
    ("provider", "prefix", "default_base_url"),
    [
        ("glm", "GLM", "https://open.bigmodel.cn/api/paas/v4"),
        ("deepseek", "DEEPSEEK", "https://api.deepseek.com"),
    ],
)
def test_provider_configuration_is_loaded_from_environment(
    provider,
    prefix,
    default_base_url,
):
    """两类供应商应使用各自的环境变量和默认 base URL。"""
    config = load_provider_config(
        provider,
        {
            f"{prefix}_API_KEY": "test-secret",
            f"{prefix}_MODEL": f"{provider}-test-model",
        },
    )

    assert config.provider == provider
    assert config.api_key == "test-secret"
    assert config.model == f"{provider}-test-model"
    assert config.base_url == default_base_url
    assert "test-secret" not in repr(config)


def test_missing_provider_configuration_returns_chinese_error():
    """缺少密钥或模型名称时，应报告缺失变量而不是发起请求。"""
    with pytest.raises(ModelConfigurationError, match="GLM_API_KEY"):
        load_provider_config("glm", {})


def test_adapter_sends_messages_and_parses_json_object():
    """适配器应把消息和模型名交给 SDK，并返回解析后的动作字典。"""
    client = FakeOpenAIClient(
        content=(
            '{"action":"submit_conclusion","root_cause":"测试根因",'
            '"evidence":[],"confidence":"low","unknowns":["待确认"]}'
        )
    )
    adapter = OpenAICompatibleModel(_config("glm"), client=client)
    messages = [
        {"role": "system", "content": "测试规则"},
        {"role": "user", "content": "测试问题"},
    ]

    result = adapter(messages)

    assert result["action"] == "submit_conclusion"
    assert client.chat.completions.requests == [
        {
            "model": "glm-test-model",
            "messages": messages,
        }
    ]


def test_adapter_parses_json_code_fence():
    """部分模型会用 Markdown 包裹 JSON，适配器应提取其中的对象。"""
    client = FakeOpenAIClient(
        content='```json\n{"action":"submit_conclusion"}\n```'
    )
    adapter = OpenAICompatibleModel(_config("deepseek"), client=client)

    result = adapter([{"role": "user", "content": "测试"}])

    assert result == {"action": "submit_conclusion"}


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "[]",
        '{"action":"call_tool"',
        "",
    ],
)
def test_adapter_rejects_non_object_model_output(content):
    """非法 JSON 和非对象结果应转为明确的模型输出错误。"""
    client = FakeOpenAIClient(content=content)
    adapter = OpenAICompatibleModel(_config(), client=client)

    with pytest.raises(ModelOutputError):
        adapter([{"role": "user", "content": "测试"}])


def test_adapter_maps_request_failure_without_exposing_secret():
    """SDK 请求异常应转换为中文错误，且异常消息不包含密钥。"""
    client = FakeOpenAIClient(error=RuntimeError("network unavailable"))
    adapter = OpenAICompatibleModel(_config(), client=client)

    with pytest.raises(ModelRequestError) as error:
        adapter([{"role": "user", "content": "测试"}])

    assert "模型请求失败" in str(error.value)
    assert "test-secret" not in str(error.value)


def test_default_client_disables_sdk_retry(monkeypatch):
    """默认 OpenAI 客户端必须关闭 SDK 自动重试，由 Agent 控制重试次数。"""
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeOpenAIClient(content='{"action":"submit_conclusion"}')

    monkeypatch.setattr("ops_agent_runtime.model_adapters.OpenAI", fake_openai)
    config = _config()

    OpenAICompatibleModel(config)

    assert captured["api_key"] == "test-secret"
    assert captured["base_url"] == config.base_url
    assert captured["max_retries"] == 0
    assert captured["timeout"] == config.timeout_seconds


def test_build_model_from_env_uses_provider_configuration():
    """工厂函数应按供应商读取环境配置并构造适配器。"""
    client = FakeOpenAIClient(content='{"action":"submit_conclusion"}')

    adapter = build_model_from_env(
        "deepseek",
        environ={
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "DEEPSEEK_MODEL": "deepseek-test-model",
        },
        client=client,
    )

    assert adapter.config.provider == "deepseek"
    assert adapter.config.model == "deepseek-test-model"


def test_provider_config_loads_dotenv_from_current_directory(monkeypatch):
    """未显式传入环境时应查找并加载 .env，同时禁止覆盖已有环境变量。"""
    calls = []
    monkeypatch.setattr(
        "ops_agent_runtime.model_adapters.find_dotenv",
        lambda usecwd: "/project/.env" if usecwd else "",
    )
    monkeypatch.setattr(
        "ops_agent_runtime.model_adapters.load_dotenv",
        lambda path, override: calls.append((path, override)),
    )
    monkeypatch.setenv("GLM_API_KEY", "process-secret")
    monkeypatch.setenv("GLM_MODEL", "process-model")

    config = load_provider_config("glm")

    assert config.api_key == "process-secret"
    assert config.model == "process-model"
    assert calls == [("/project/.env", False)]
