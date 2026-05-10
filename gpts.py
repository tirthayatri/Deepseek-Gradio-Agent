import os
import func_timeout
from func_timeout import func_set_timeout
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any

load_dotenv()


@func_set_timeout(100)
def openai_core(msg, version, temperature=0, stream=False, reasoning_effort=None, thinking=False):
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
        base_url="https://api.deepseek.com"
    )
    kwargs: dict[str, Any] = dict(
        model=version,
        messages=msg,
        temperature=temperature,
        max_tokens=8000,
        stream=stream,
    )
    if reasoning_effort is not None:
        kwargs['reasoning_effort'] = reasoning_effort
    if thinking:
        kwargs['extra_body'] = {"thinking": {"type": "enabled"}}
    return client.chat.completions.create(**kwargs)


def call_openai(sys, user, version='deepseek-v4-flash', temperature=0, max_test_num=3):
    msg = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    idx = 0
    while idx < max_test_num:
        idx += 1
        try:
            response = openai_core(msg, version, temperature)  # type: ignore[call-arg]
            return response.choices[0].message.content  # type: ignore[union-attr]
        except func_timeout.exceptions.FunctionTimedOut as e:
            print(e)
            print('single gpt request time out!')
    return None
