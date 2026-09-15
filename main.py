"""掌柜智库统一入口：python main.py import|query|all"""
import argparse
import multiprocessing
import sys


def _setup_console() -> None:
    """Windows 中文控制台为 GBK，手册中的 © 等字符会使 print 抛
    UnicodeEncodeError 炸掉节点流程；统一将标准流切到 UTF-8 并容错。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _run(service: str) -> None:
    _setup_console()
    if service == "import":
        from web.api.import_service import app

        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        from web.api.query_service import app

        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8001)


def main() -> None:
    parser = argparse.ArgumentParser(description="掌柜智库服务启动器")
    parser.add_argument(
        "service",
        choices=["import", "query", "all"],
        nargs="?",
        default="all",
        help="启动导入服务(8000)/查询服务(8001)/全部",
    )
    args = parser.parse_args()

    targets = ["import", "query"] if args.service == "all" else [args.service]
    processes = [multiprocessing.Process(target=_run, args=(t,), name=t) for t in targets]
    for p in processes:
        p.start()
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
