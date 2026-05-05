import os
import json
import datetime
import gradio as gr
import tiktoken
import requests
from dotenv import load_dotenv
from gpts import openai_core, call_openai

load_dotenv()

SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY', '')
HISTORY_DIR = 'history'


def cal_token_num(history):
    content = ''.join(c['content'] for c in history)
    return len(tiktoken.get_encoding('cl100k_base').encode(content))


def predict_block(system, system1, user_radio, history, memory_num, version, temperature):
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise gr.Error("未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置后重启。")

    import random
    if random.random() > 1 - user_radio:
        cont = f"你要扮演一个角色，人格描述是：{system}"
    else:
        cont = f"你要扮演一个角色，人格描述是：{system}。下面将要和你对话的朋友，人格描述为：{system1}。"

    history_to_use = history[-int(memory_num):] if len(history) > memory_num else history
    history_openai_format = [{"role": "system", "content": cont}]

    for human, assistant in history_to_use:
        history_openai_format.append({"role": "user", "content": human})
        if assistant is not None:
            history_openai_format.append({"role": "assistant", "content": assistant})

    try:
        response = openai_core(history_openai_format, version, temperature, stream=True)
    except Exception as e:
        raise gr.Error(f"API 请求失败：{str(e)}")

    partial_message = ""
    for chunk in response:
        try:
            tmp = chunk.choices[0].delta.content
        except Exception:
            tmp = ''
        if tmp:
            partial_message += tmp
        history[-1][1] = partial_message
        yield history, cal_token_num(history_openai_format)


def predict_block_direct(system, history, memory_num, version, temperature):
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise gr.Error("未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置后重启。")

    history_to_use = history[-int(memory_num):] if len(history) > memory_num else history
    history_openai_format = [{"role": "system", "content": system}]

    for human, assistant in history_to_use:
        history_openai_format.append({"role": "user", "content": human})
        if assistant is not None:
            history_openai_format.append({"role": "assistant", "content": assistant})

    try:
        response = openai_core(history_openai_format, version, temperature)
        text = response.choices[0].message.content
    except Exception as e:
        raise gr.Error(f"API 请求失败：{str(e)}")

    try:
        output_file = text_to_audio(text)
    except Exception as e:
        gr.Warning(f"语音合成失败：{str(e)}")
        output_file = None

    history[-1][1] = text
    return history, output_file


def user(user_message, history):
    return "", history + [[user_message, None]]


def auth(usr, pwd):
    with open("auth.json", 'r') as s:
        d = json.load(s)
        if usr in d['gpt']:
            return d['gpt'][usr] == pwd
    return False


def load_model_sorted(path):
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def load_personas(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def change_persona(key, path):
    return load_personas(path)[key]


def save_persona(key, value, path):
    data = load_personas(path)
    data[key] = value
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def make_persona(persona, user_persona, version, temperature):
    sys = (
        f"你扮演一个角色，人格描述是：{persona}，从你的眼光判断，并合理发散幻想一下，下面你这位朋友的人格描述。"
        "请用一整段文字进行回复，不要出现1.2.3的条目。"
        "请检查基本信息，不要与客观基本信息有任何出入。"
        "请尽量详细的设想这个人的其他方面，甚至包括他可能喜欢什么讨厌什么，回复尽量详细。"
    )
    user_msg = f"这个人的基本信息为：{user_persona}"
    return call_openai(sys, user_msg, version, temperature)


def audio_to_text(filepath, history):
    if not SILICONFLOW_API_KEY:
        raise gr.Error("未设置 SILICONFLOW_API_KEY，语音功能不可用。")
    try:
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
        with open(filepath, "rb") as f:
            response = requests.post(
                "https://api.siliconflow.cn/v1/audio/transcriptions",
                headers=headers,
                files={"file": f},
                data={"model": "FunAudioLLM/SenseVoiceSmall"}
            )
        response.raise_for_status()
        user_message = response.json()['text']
    except Exception as e:
        raise gr.Error(f"语音识别失败：{str(e)}")
    return None, history + [[user_message, None]]


def text_to_audio(text):
    if not SILICONFLOW_API_KEY:
        gr.Warning("未设置 SILICONFLOW_API_KEY，跳过语音合成。")
        return None
    output_file = 'output.wav'
    payload = {
        "model": "fishaudio/fish-speech-1.5",
        "input": text,
        "voice": "fishaudio/fish-speech-1.5:alex",
        "response_format": "wav",
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.siliconflow.cn/v1/audio/speech", json=payload, headers=headers)
    response.raise_for_status()
    with open(output_file, "wb") as wav_file:
        wav_file.write(response.content)
    return output_file


def list_histories():
    if not os.path.exists(HISTORY_DIR):
        return []
    return sorted(
        [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')],
        reverse=True
    )


def save_history(history, name):
    if not history:
        gr.Warning("当前对话为空，无需保存。")
        return gr.update()
    os.makedirs(HISTORY_DIR, exist_ok=True)
    filename = (name.strip() or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.json'
    with open(os.path.join(HISTORY_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    gr.Info(f"已保存：{filename}")
    return gr.update(choices=list_histories(), value=filename)


def load_history_file(filename):
    if not filename:
        gr.Warning("请先选择要加载的对话记录。")
        return gr.update()
    with open(os.path.join(HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 8px 16px !important;
    }
    /* Left chatbot fills viewport height */
    #chatbot {
        min-height: calc(100vh - 80px) !important;
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
    }
    /* Input box */
    #msg-input textarea {
        font-size: 14px !important;
        border-radius: 8px !important;
    }
    /* All buttons: unified size */
    .btn-u {
        min-height: 40px !important;
        height: 40px !important;
        font-size: 14px !important;
        border-radius: 6px !important;
    }
    /* Header title */
    #page-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 2.2 !important;
    }
    """

    os.makedirs(HISTORY_DIR, exist_ok=True)
    dropdown_options = load_model_sorted('models.txt')
    personas = load_personas('persona.json')
    personas_keys = list(personas.keys())
    user_personas = load_personas('user.json')
    user_personas_keys = list(user_personas.keys())

    with gr.Blocks(css=custom_css, title="DeepSeek Chat") as demo:

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    show_copy_button=True,
                    elem_id="chatbot",
                    bubble_full_width=False,
                    label="对话",
                )

            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("**DeepSeek Chat**", elem_id="page-title")
                    model = gr.Dropdown(
                        value=dropdown_options[0], choices=dropdown_options,
                        label="模型", scale=0, min_width=190,
                    )

                msg = gr.Textbox(
                    placeholder="输入消息，Enter 发送…",
                    show_label=False,
                    container=False,
                    elem_id="msg-input",
                    lines=3,
                    autofocus=True,
                )

                with gr.Row():
                    send_btn = gr.Button(
                        "发送", variant="primary", scale=1,
                        elem_classes=["btn-u"],
                    )
                    clear_btn = gr.Button(
                        "清空", variant="stop", scale=1,
                        elem_classes=["btn-u"],
                    )

                token_num = gr.Number(value=0, visible=False, interactive=False)

                with gr.Accordion("参数设置", open=False):
                    with gr.Row():
                        memory_num = gr.Number(value=10, label="记忆轮数", interactive=True)
                        temperature = gr.Number(value=0.7, label="温度", interactive=True)
                        user_ratio = gr.Number(value=0.1, label="用户视角比例", interactive=True)

                with gr.Accordion("角色设置", open=False):
                    with gr.Tabs():
                        with gr.Tab("AI 角色"):
                            persona_opt = gr.Dropdown(
                                value=personas_keys[0], choices=personas_keys,
                                label="选择角色",
                            )
                            persona_txt = gr.Textbox(
                                label="角色描述", lines=5,
                                value=personas[personas_keys[0]],
                            )
                            persona_save = gr.Button(
                                "保存修改", elem_classes=["btn-u"],
                            )
                        with gr.Tab("用户角色"):
                            user_persona_opt = gr.Dropdown(
                                value=user_personas_keys[0], choices=user_personas_keys,
                                label="选择用户",
                            )
                            user_persona_txt = gr.Textbox(
                                label="用户描述", lines=5,
                                value=user_personas[user_personas_keys[0]],
                            )
                            user_persona_save = gr.Button(
                                "保存修改", elem_classes=["btn-u"],
                            )

                with gr.Accordion("对话历史", open=False):
                    with gr.Row():
                        save_name = gr.Textbox(
                            placeholder="对话名称（留空自动时间戳）",
                            show_label=False, container=False, scale=3,
                        )
                        save_btn = gr.Button(
                            "保存", scale=1, min_width=72,
                            elem_classes=["btn-u"],
                        )
                    with gr.Row():
                        history_dropdown = gr.Dropdown(
                            choices=list_histories(), show_label=False,
                            container=False, scale=3,
                        )
                        load_btn = gr.Button(
                            "加载", scale=1, min_width=72,
                            elem_classes=["btn-u"],
                        )
                        refresh_btn = gr.Button(
                            "刷新", scale=1, min_width=72,
                            elem_classes=["btn-u"],
                        )

        with gr.Row(visible=False):
            rec = gr.Audio(sources=['microphone'], type="filepath")
            rec_btn = gr.Button("提交语音")
            rec_resp = gr.Audio(type="filepath")

        gen_inputs = [persona_txt, user_persona_txt, user_ratio, chatbot, memory_num, model, temperature]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            predict_block, gen_inputs, [chatbot, token_num]
        )
        send_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            predict_block, gen_inputs, [chatbot, token_num]
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

        persona_opt.change(lambda x: change_persona(x, 'persona.json'), [persona_opt], [persona_txt])
        persona_save.click(lambda x, y: save_persona(x, y, 'persona.json'), [persona_opt, persona_txt])
        user_persona_opt.change(lambda x: change_persona(x, 'user.json'), [user_persona_opt], [user_persona_txt])
        user_persona_save.click(lambda x, y: save_persona(x, y, 'user.json'), [user_persona_opt, user_persona_txt])

        save_btn.click(save_history, [chatbot, save_name], [history_dropdown])
        load_btn.click(load_history_file, [history_dropdown], [chatbot])
        refresh_btn.click(lambda: gr.update(choices=list_histories()), outputs=[history_dropdown])

        rec_btn.click(
            audio_to_text, [rec, chatbot], [rec, chatbot]
        ).then(
            predict_block_direct,
            [persona_txt, chatbot, memory_num, model, temperature],
            [chatbot, rec_resp]
        )

    demo.queue()
    demo.launch(server_port=19001, show_api=False, root_path="/gpt")
