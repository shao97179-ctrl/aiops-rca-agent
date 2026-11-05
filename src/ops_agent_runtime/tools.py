import json
from datetime import datetime
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "01_bad_config_deploy"
METRICS_FILE = SCENARIO_DIR / "metrics.json"
LOGS_FILE = SCENARIO_DIR / "logs.jsonl"


def _normalize_time(value, name):
    """把时间统一成 YYYY-MM-DDTHH:MM；不合法时返回 (None, 错误 dict)。"""
    if value is None:
        return None, None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%dT%H:%M"), None
    except (ValueError, TypeError):
        return None, {
            "error": f"invalid {name}: {value!r}, expected format YYYY-MM-DDTHH:MM"
        }


def query_metrics(metric, scope, start=None, end=None):
    """按指标名、作用域和时间范围查询指标序列。"""
    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"failed to read metrics data: {exc}"}

    series_list = data["series"]
    available_metrics = sorted({s["name"] for s in series_list})
    available_scopes = sorted({s["scope"] for s in series_list})

    if metric not in available_metrics:
        return {
            "error": f"unknown metric: {metric!r}",
            "available_metrics": available_metrics,
            "available_scopes": available_scopes,
        }
    if scope not in available_scopes:
        return {
            "error": f"unknown scope: {scope!r}",
            "available_metrics": available_metrics,
            "available_scopes": available_scopes,
        }

    start_ts, error = _normalize_time(start, "start")
    if error:
        return error
    end_ts, error = _normalize_time(end, "end")
    if error:
        return error
    if start_ts and end_ts and start_ts > end_ts:
        return {"error": f"start ({start_ts}) is after end ({end_ts})"}

    matched = [s for s in series_list if s["name"] == metric and s["scope"] == scope]
    if not matched:
        return {
            "error": f"no series for metric={metric!r} scope={scope!r}",
            "available_series": [
                {"metric": s["name"], "scope": s["scope"]} for s in series_list
            ],
        }

    series = matched[0]
    points = [
        p
        for p in series["points"]
        if (start_ts is None or p[0] >= start_ts) and (end_ts is None or p[0] <= end_ts)
    ]

    all_ts = [p[0] for p in series["points"]]
    result = {
        "metric": series["name"],
        "scope": series["scope"],
        "unit": series.get("unit"),
        "window": {
            "start": start_ts or (min(all_ts) if all_ts else None),
            "end": end_ts or (max(all_ts) if all_ts else None),
        },
        "points": points,
    }
    if not points:
        result["note"] = "no data points in the requested time range"
    return result


def query_logs(service=None, level=None, start=None, end=None):
    """按服务、级别和时间范围查询日志，结束时间包含所选整分钟。"""
    try:
        with open(LOGS_FILE, encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"failed to read log data: {exc}"}

    available_services = sorted({log["service"] for log in logs})
    available_levels = sorted({log["level"] for log in logs})

    if service is not None and service not in available_services:
        return {
            "error": f"unknown service: {service!r}",
            "available_services": available_services,
            "available_levels": available_levels,
        }
    if level is not None and level not in available_levels:
        return {
            "error": f"unknown level: {level!r}",
            "available_services": available_services,
            "available_levels": available_levels,
        }

    start_ts, error = _normalize_time(start, "start")
    if error:
        return error
    end_ts, error = _normalize_time(end, "end")
    if error:
        return error
    if start_ts and end_ts and start_ts > end_ts:
        return {"error": f"start ({start_ts}) is after end ({end_ts})"}

    matched = [
        log
        for log in logs
        if (service is None or log["service"] == service)
        and (level is None or log["level"] == level)
        and (
            start_ts is None
            or log["ts"][:16] >= start_ts
        )
        and (
            end_ts is None
            or log["ts"][:16] <= end_ts
        )
    ]
    result = {"logs": matched, "total": len(matched)}
    if not matched:
        result["note"] = "no log entries matched the requested filters"
    return result
