# Deepseek-Gradio-Agent

一个基于Gradio的多角色AI聊天机器人，支持自定义人格、语音输入/输出

## 功能特性

- **角色扮演**：内置多套AI人格，且支持自定义人格
- **用户人格**：为对话对象设置人格描述，影响AI的回复风格
- **语音交互**：支持麦克风输入和TTS语音输出
- **对话记忆**：可调节记忆轮数，控制上下文长度

## 快速开始

### 1. 克隆仓库

**HTTPS：**
```bash
git clone https://github.com/tirthayatri/Deepseek-Gradio-Agent.git
cd Deepseek-Gradio-Agent
```

**SSH：**
```bash
git clone git@github.com:tirthayatri/Deepseek-Gradio-Agent.git
cd Deepseek-Gradio-Agent
```

### 2. 创建并激活虚拟环境

```bash
python -m venv venv
```

**Windows：**
```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux：**
```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API 密钥

参考`.env.example`在项目根目录下新建一个名为 `.env` 的文件：

```
DEEPSEEK_API_KEY=你的密钥
SILICONFLOW_API_KEY=你的密钥
```

> **说明**
> - `.env.example` 是模板，展示了需要填写哪些变量，不要直接修改它
> - 新建的 `.env` 文件已被 `.gitignore` 排除
> - `SILICONFLOW_API_KEY` 仅语音功能需要填写

获取密钥：
- DeepSeek 密钥：[platform.deepseek.com](https://platform.deepseek.com)
- SiliconFlow 密钥：[cloud.siliconflow.cn](https://cloud.siliconflow.cn)

### 5. 运行

```bash
python grdio_ui.py
```

浏览器访问 `http://localhost:19001`

## 文件说明

| 文件 | 说明 |
|------|------|
| `grdio_ui.py` | 主程序，Gradio UI 和业务逻辑 |
| `gpts.py` | DeepSeek API 封装 |
| `persona.json` | AI 角色人格配置 |
| `user.json` | 用户人格配置 |
| `models.txt` | 可用模型列表 |
| `region.txt` | 搜索区域列表 |

## 自定义人格

在界面底部的Persona面板中编辑，或手动编辑 `persona.json`。
