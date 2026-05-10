"""
将 stdout 同时输出到终端和日志文件的工具。

用法：
    from _logger import log_to_file

    with log_to_file("success_rate"):
        run(n)
"""

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

LOGS_DIR = Path(__file__).parent / "logs"


class _Tee:
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)
        # \r（进度覆写）在日志里转为换行，\r\n 先合并避免双换行
        self._file.write(data.replace("\r\n", "\n").replace("\r", "\n"))
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()


@contextmanager
def log_to_file(prefix: str) -> Generator[Path, None, None]:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{prefix}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = _Tee(f)  # type: ignore[assignment]
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout

    print(f"日志已保存：{log_path}")
