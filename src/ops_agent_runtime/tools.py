import json
from datetime import datetime
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "01_bad_config_deploy"
METRICS_FILE = SCENARIO_DIR / "metrics.json"
LOGS_FILE = SCENARIO_DIR / "logs.jsonl"
DEPLOYS_FILE = SCENARIO_DIR / "deploys.json"


def _normalize_time(value, name):
    """把时间统一成 YYYY-MM-DDTHH:MM；不合法时返回 (None, 错误 dict)。"""
    if value is None:
        return None, None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%dT%H:%M"), None
    except (ValueError, TypeError):
        return None, {
            "error": f"{name}不合法：{value!r}，时间格式应为 YYYY-MM-DDTHH:MM"
        }


def _normalize_range(start, end):
    """校验查询时间范围，返回 (start_ts, end_ts, error)。"""
    start_ts, error = _normalize_time(start, "开始时间")
    if error:
        return None, None, error
    end_ts, error = _normalize_time(end, "结束时间")
    if error:
        return None, None, error
    if start_ts and end_ts and start_ts > end_ts:
        return None, None, {
            "error": f"开始时间 {start_ts} 晚于结束时间 {end_ts}"
        }
    return start_ts, end_ts, None


def query_metrics(metric, scope, start=None, end=None):
    """按指标名、作用域和时间范围查询指标序列。"""
    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"读取指标数据失败：{exc}"}

    series_list = data["series"]
    available_metrics = sorted({s["name"] for s in series_list})
    available_scopes = sorted({s["scope"] for s in series_list})

    if metric not in available_metrics:
        return {
            "error": f"未知指标：{metric!r}",
            "available_metrics": available_metrics,
            "available_scopes": available_scopes,
        }
    if scope not in available_scopes:
        return {
            "error": f"未知作用域：{scope!r}",
            "available_metrics": available_metrics,
            "available_scopes": available_scopes,
        }

    start_ts, end_ts, error = _normalize_range(start, end)
    if error:
        return error

    matched = [s for s in series_list if s["name"] == metric and s["scope"] == scope]
    if not matched:
        return {
            "error": f"未找到指标 {metric!r}、作用域 {scope!r} 对应的数据序列",
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
        result["note"] = "未找到与请求的时间范围相匹配的指标数据"
    return result


def query_logs(service=None, level=None, start=None, end=None):
    """按服务、级别和时间范围查询日志，结束时间包含所选整分钟。"""
    try:
        with open(LOGS_FILE, encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"读取日志数据失败：{exc}"}

    available_services = sorted({log["service"] for log in logs})
    available_levels = sorted({log["level"] for log in logs})

    if service is not None and service not in available_services:
        return {
            "error": f"未知服务：{service!r}",
            "available_services": available_services,
            "available_levels": available_levels,
        }
    if level is not None and level not in available_levels:
        return {
            "error": f"未知日志级别：{level!r}",
            "available_services": available_services,
            "available_levels": available_levels,
        }

    start_ts, end_ts, error = _normalize_range(start, end)
    if error:
        return error

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
        result["note"] = "未找到与请求的筛选条件相匹配的日志条目"
    return result


def query_deploys(service=None, start=None, end=None):
    """查询发布记录，结束时间包含所选整分钟。"""
    try:
        with open(DEPLOYS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        deploys = data["deploys"]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"读取发布数据失败：{exc}"}

    available_services = sorted({deploy["service"] for deploy in deploys})
    if service is not None and service not in available_services:
        return {
            "error": f"未知服务：{service!r}",
            "available_services": available_services,
        }

    start_ts, end_ts, error = _normalize_range(start, end)
    if error:
        return error

    matched = [
        deploy
        for deploy in deploys
        if (service is None or deploy["service"] == service)
        and (
            start_ts is None
            or deploy["time"][:16] >= start_ts
        )
        and (
            end_ts is None
            or deploy["time"][:16] <= end_ts
        )
    ]
    result = {"deploys": matched, "total": len(matched)}
    if not matched:
        result["note"] = "未找到与请求的筛选条件相匹配的发布记录"
    return result
