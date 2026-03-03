# 智能代理助手（ReAct Agent）

这是一个基于 ReAct（Reasoning + Action）框架构建的智能代理系统，能够理解用户指令、自主调用工具（如搜索、计算、查询等），并通过多轮推理完成复杂任务。

## 核心特性

- **自主推理与行动**：模型可生成 `Thought → Action → Observation` 循环，逐步解决问题。
- **工具扩展性强**：支持灵活注册自定义工具（如网络搜索、天气查询、代码执行等）。
- **兼容主流大模型**：通过 OpenAI 兼容接口，支持通义千问（Qwen）、DeepSeek、Ollama 等多种 LLM。
- **安全可控**：限制最大推理步数（默认 5 步），防止无限循环。

## 快速开始

### 1. 安装依赖

pip install -r requirements.txt

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：
```env
API_KEY=your_dashscope_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. 运行示例

```python
from agent import run

# 示例：查询天气并总结
run("今天北京天气如何？适合出门吗？")
```
程序将自动：
1. 分析用户意图
2. 调用 `get_weather` 工具获取实时天气
3. 结合天气信息生成自然语言回答

### 配置说明

```markdown
| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `API_KEY` | Qwen API 密钥 | 从 `.env` 读取 |
| `BASE_URL` | LLM 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_ID` | 使用的模型名称 | `qwen3.5-plus` |
| `TAVILY_API_KEY` | Tavily 搜索 API（用于 `web_search` 工具） | 可选 |
你可以轻松切换为其他 OpenAI 兼容服务（如 Ollama、DeepSeek、Moonshot 等），只需修改 `BASE_URL` 和 `MODEL_ID`。
```

## 扩展建议

- 添加更多工具：股票查询、日历、计算器、数据库访问等
- 支持多模态输入（图像、语音）
- 增加记忆模块（长期上下文）
- 部署为 Web 服务（FastAPI / Flask）

> **提示**：本代理遵循严格的 `Thought/Action/Observation` 格式，请确保你的 LLM 提示词（`AGENT_SYSTEM_PROMPT`）明确要求此结构。