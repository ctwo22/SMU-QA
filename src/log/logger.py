"""极简运维日志：同时写 stderr 和按大小轮转的文件，统一 UTF-8。

用 stream=stderr 是硬要求：MCP stdio 传输里 stdout 是 JSON-RPC 协议通道，
任何多余的 stdout 输出都会让客户端解析失败。
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 固定落在项目根的 logs/，不随启动时的工作目录漂移
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_FORMAT = logging.Formatter(
    "%(asctime)s %(levelname)-7s %(process)d %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _force_utf8_stderr() -> None:
    """把 stderr 切到 UTF-8，中文和 banner 的方块字符才不会乱码。

    子进程由 stdio_client 拉起，只继承 PATH 等少数环境变量（见 mcp 的
    DEFAULT_INHERITED_ENV_VARS），IDE 里设的 PYTHONIOENCODING 传不进来，
    于是子进程按系统代码页 GBK 输出 —— 这就是乱码的根因，所以要在代码里设。
    只动 stderr，不碰 stdout：stdout 是 MCP 协议通道，由传输层自己接管。
    """
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError, OSError):
        pass  # 被替换成非标准流时静默跳过，不影响功能


def setup_logging(name: str, filename: str, level: str | None = None) -> logging.Logger:
    """配置并返回 logger。重复调用不会重复挂 handler。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False  # 不向 root 传播，避免每行打印两遍

    _force_utf8_stderr()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_FORMAT)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)

    # 受限终端 / 只读目录里打不开日志文件时，
    # 只保留 stderr 输出，绝不能让"写不了日志"把整个服务拖垮。
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_DIR / filename,
            maxBytes=5 * 1024 * 1024,  # 5MB 轮转，保留 3 份
            backupCount=3,
            encoding="utf-8",  # 不加这个，中文会被 GBK 转义成 \uXXXX
        )
        file_handler.setFormatter(_FORMAT)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(
            "文件日志不可用（%s）：%s；本次只输出到 stderr", LOG_DIR / filename, exc
        )

    return logger
