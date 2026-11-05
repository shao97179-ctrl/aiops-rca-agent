from ops_agent_runtime.tools import query_metrics
from ops_agent_runtime import tools

def test_range_boundaries_are_inclusive():
    result = query_metrics(
        metric="error_rate",
        scope="global",
        start="2026-11-01T12:00",
        end="2026-11-01T12:10",
    )

    assert result["points"] == [
        ["2026-11-01T12:00", 0.3],
        ["2026-11-01T12:05", 38.6],
        ["2026-11-01T12:10", 41.2],
    ]


def test_out_of_range_returns_empty_result():
    result = query_metrics(
        metric="error_rate",
        scope="global",
        start="2026-11-01T13:00",
        end="2026-11-01T13:10",
    )

    assert "error" not in result
    assert result["points"] == []
    assert result["window"] == {
        "start": "2026-11-01T13:00",
        "end": "2026-11-01T13:10",
    }
    assert result["note"] == "未找到与请求的时间范围相匹配的指标数据"


def test_unknown_metric_returns_available_metrics():
    result = query_metrics(metric="latency_p99", scope="global")

    assert "error" in result
    assert result["available_metrics"] == ["error_rate", "latency_p95"]


def test_unknown_scope_returns_available_scopes():
    result = query_metrics(metric="error_rate", scope="database")

    assert "error" in result
    assert result["available_scopes"] == ["cart", "global", "payment"]


def test_start_after_end_returns_error():
    result = query_metrics(
        metric="error_rate",
        scope="global",
        start="2026-11-01T12:10",
        end="2026-11-01T12:00",
    )

    assert "error" in result
    assert result["error"] == (
        "start (2026-11-01T12:10) is after end (2026-11-01T12:00)"
    )

def test_missing_metrics_file_returns_error(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing.json"
    monkeypatch.setattr(tools, "METRICS_FILE", missing_file)

    result = tools.query_metrics(
        metric="error_rate",
        scope="global",
    )

    assert "error" in result
    assert result["error"].startswith("failed to read metrics data")


def test_invalid_json_returns_error(monkeypatch, tmp_path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{this is not valid json", encoding="utf-8")
    monkeypatch.setattr(tools, "METRICS_FILE", invalid_file)

    result = tools.query_metrics(
        metric="error_rate",
        scope="global",
    )

    assert "error" in result
    assert result["error"].startswith("failed to read metrics data")

def test_query_logs_filters_service_level_and_minute():
    result = tools.query_logs(
        service="payment",
        level="ERROR",
        start="2026-11-01T12:05",
        end="2026-11-01T12:05",
    )

    assert "error" not in result
    assert result["total"] == 4
    assert [log["id"] for log in result["logs"]] == [
        "log_0011",
        "log_0012",
        "log_0013",
        "log_0015",
    ]


def test_query_logs_unknown_service_returns_available_values():
    result = tools.query_logs(service="database")

    assert "error" in result
    assert result["available_services"] == ["cart", "frontend", "payment", "user"]
    assert result["available_levels"] == ["ERROR", "INFO", "WARN"]
    assert "logs" not in result


def test_query_logs_unknown_level_returns_available_values():
    result = tools.query_logs(level="FATAL")

    assert "error" in result
    assert result["available_levels"] == ["ERROR", "INFO", "WARN"]
    assert "logs" not in result


def test_query_logs_no_matches_is_not_an_error():
    result = tools.query_logs(service="cart", level="ERROR")

    assert "error" not in result
    assert result["logs"] == []
    assert result["total"] == 0
    assert result["note"] == "未找到与请求的筛选条件相匹配的日志条目"


def test_missing_log_file_returns_error(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing.jsonl"
    monkeypatch.setattr(tools, "LOGS_FILE", missing_file)

    result = tools.query_logs()

    assert "error" in result
    assert result["error"].startswith("failed to read log data")


def test_query_deploys_filters_service_and_minute():
    result = tools.query_deploys(
        service="payment",
        start="2026-11-01T12:00",
        end="2026-11-01T12:00",
    )

    assert "error" not in result
    assert result["total"] == 1
    assert result["deploys"][0]["id"] == "dep_d1003"
    assert result["deploys"][0]["detail"] == {
        "file": "payment/config.yaml",
        "modified_key": "dependency.cart.port",
        "old_value": "8080",
        "new_value": "8081",
    }


def test_query_deploys_unknown_service_returns_available_services():
    result = tools.query_deploys(service="database")

    assert "error" in result
    assert result["available_services"] == ["cart", "frontend", "payment"]
    assert "deploys" not in result


def test_query_deploys_no_matches_is_not_an_error():
    result = tools.query_deploys(
        service="payment",
        start="2026-11-01T13:00",
        end="2026-11-01T13:10",
    )

    assert "error" not in result
    assert result["deploys"] == []
    assert result["total"] == 0
    assert result["note"] == "未找到与请求的筛选条件相匹配的发布记录"


def test_missing_deploy_file_returns_error(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing.json"
    monkeypatch.setattr(tools, "DEPLOYS_FILE", missing_file)

    result = tools.query_deploys()

    assert "error" in result
    assert result["error"].startswith("failed to read deploy data")
