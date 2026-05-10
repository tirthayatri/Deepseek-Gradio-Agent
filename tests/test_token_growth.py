"""
测量多轮对话下的 Token 累计消耗，验证 N 轮对话是否会触达上下文窗口上限。
    python tests/test_token_growth.py [ROUNDS]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import tiktoken

from gpts import openai_core

load_dotenv()

CONTEXT_LIMIT = 1_000_000      # DeepSeek V4 1M 上下文
OUTPUT_RESERVE = 384_000       # DeepSeek V4 最大输出 384K
USABLE = CONTEXT_LIMIT - OUTPUT_RESERVE
MODEL = "deepseek-v4-flash"
DEFAULT_ROUNDS = 20            #change this

USER_TURNS = [
    "你最近在玩什么游戏？", "推荐一款机械键盘", "你怎么看 AI 编程？",
    "周末有什么计划", "聊聊《终结者》", "Python 和 Rust 你更喜欢哪个",
    "今天天气怎么样", "怎么学好算法", "推荐一部纪录片", "Vim 还是 VSCode",
    "你怎么看 996", "周末想去爬山，建议", "最近有什么好电影",
    "怎么提升英语", "聊聊量子计算", "推荐一款相机", "怎么坚持健身",
    "你最喜欢的编程语言", "AI 会取代程序员吗", "再见",
]

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict]) -> int:
    return sum(len(enc.encode(m["content"])) for m in messages)


def run(rounds: int = DEFAULT_ROUNDS) -> list[dict]:
    system = "你是一个性格外向、热爱技术的程序员，回复请控制在 100 字以内。"
    history = [{"role": "system", "content": system}]
    results = []

    print(f"{'轮次':<6}{'输入tokens':<14}{'累计tokens':<14}{'占用率':<10}")
    print("-" * 44)

    for i in range(rounds):
        user_msg = USER_TURNS[i % len(USER_TURNS)]
        history.append({"role": "user", "content": user_msg})
        in_tok = count_tokens(history)

        resp = openai_core(history, version=MODEL, temperature=0.7)  # type: ignore[call-arg]
        reply = resp.choices[0].message.content  # type: ignore[union-attr]
        history.append({"role": "assistant", "content": reply})

        total = count_tokens(history)
        pct = total / USABLE * 100
        results.append({"round": i + 1, "input": in_tok, "total": total, "pct": pct})
        print(f"{i + 1:<6}{in_tok:<14}{total:<14}{pct:.2f}%")

    final = results[-1]
    print()
    print("=" * 44)
    print(f"{rounds} 轮后累计 {final['total']} tokens，占可用上下文 {final['pct']:.2f}%")
    print(f"是否超限：{'是' if final['total'] > USABLE else '否'}（可用 {USABLE} tokens）")

    return results


if __name__ == "__main__":
    from _logger import log_to_file
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROUNDS
    with log_to_file("token_growth"):
        run(rounds)
