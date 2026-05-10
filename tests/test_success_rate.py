"""
本程序用于测试DeepSeek API端到端成功率与延迟
python tests/test_success_rate.py [N]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import func_timeout

from gpts import openai_core

load_dotenv()

DEFAULT_N = 200 #change this
MODEL = "deepseek-v4-flash"

PROMPTS = [
    "用一句话介绍你自己",
    "1+1 等于几？",
    "推荐一本科幻小说",
    "解释什么是 HTTP",
    "写一首四行小诗，主题是夏天",
    "Python 装饰器是什么",
    "你最喜欢的颜色是什么",
    "怎么学好算法",
]


def run(n: int = DEFAULT_N) -> dict:
    success = 0
    durations: list[float] = []
    fails = {"timeout": 0, "api_error": 0, "empty_response": 0}

    for i in range(n):
        msg = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": PROMPTS[i % len(PROMPTS)]},
        ]
        t0 = time.time()
        try:
            resp = openai_core(msg, version=MODEL, temperature=0.7)  # type: ignore[call-arg]
            content = resp.choices[0].message.content  # type: ignore[union-attr]
            if content and content.strip():
                success += 1
                durations.append(time.time() - t0)
            else:
                fails["empty_response"] += 1
        except func_timeout.exceptions.FunctionTimedOut:
            fails["timeout"] += 1
        except Exception as e:
            fails["api_error"] += 1
            print(f"\n[{i}] {type(e).__name__}: {e}")

        rate = success / (i + 1) * 100
        print(f"[{i + 1}/{n}] 成功率 {rate:.2f}%", end="\r", flush=True)

    print()
    print("=" * 40)
    print(f"成功率：{success}/{n} = {success / n * 100:.2f}%")
    print(f"失败分布：{fails}")
    if durations:
        sorted_d = sorted(durations)
        avg = sum(durations) / len(durations)
        p50 = sorted_d[len(sorted_d) // 2]
        p95 = sorted_d[int(len(sorted_d) * 0.95)]
        print(f"延迟：avg={avg:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s")

    return {"success": success, "n": n, "fails": fails, "durations": durations}


if __name__ == "__main__":
    from _logger import log_to_file
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    with log_to_file("success_rate"):
        run(n)
