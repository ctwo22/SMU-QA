"""统一启动入口：所有服务都从这里拉起。

在项目根目录执行：
    python app.py                  # 默认 127.0.0.1:8000
    python app.py --port 8080      # 换端口
    python app.py --reload         # 改代码自动重启（注意：会清空会话历史）
    python app.py --host 0.0.0.0   # 允许同一局域网的手机/其他电脑访问

也可以用环境变量覆盖默认值：HOST / PORT
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证从任何目录启动都能 import src.*
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动三明学院校园问答服务")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="监听地址，默认 127.0.0.1（仅本机）；局域网访问用 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="监听端口，默认 8000",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="代码改动后自动重启（会清空内存里的会话历史）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 切到项目根：--reload 时 uvicorn 会起子进程，靠 cwd 找得到 src 包
    os.chdir(ROOT)

    print(f"\n  Server: http://{args.host}:{args.port}\n", flush=True)

    if args.reload:
        # reload 模式必须传 "模块:变量" 字符串，uvicorn 才能监控文件变化
        uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=True)
    else:
        # 直接传 app 对象，少一次模块导入，启动更快
        from src.api.main import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
