# 三明学院 Q&A Agent

> 基于 **FastAPI + MCP + RAG** 的校园问答助手：DeepSeek 负责对话，MCP 协议调用本地校园知识库工具，
> Chroma 向量检索保证回答有据可查，前端为原生 HTML/CSS/JS，一条命令启动。

## 功能

- **校园知识问答**：当前仅关于SMU的少量资料【主要针对新生】(主要内容为作者(学长)自行整理和入党、入伍有关)，如有需要其他资料，请自行接入
- **RAG 检索**：Chroma 向量库 + 通义 Embedding；检索不到相关内容时明确返回“无相关资料”，减少模型幻觉
- **MCP 工具调用**：对话模型通过 MCP（stdio 传输）调用 `campus_search` 工具 ，作者未加入联网搜索工具， 如有需要请在mcp.server.py 中自行添加
- **会话保持**：对话历史保存在服务端，刷新页面自动恢复；支持“新建会话”清空历史

- **本地部署**：单机单用户，无需前端构建，`python app.py` 即可运行

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn |
| 对话模型 | DeepSeek Chat（OpenAI 兼容接口） |
| 工具协议 | MCP（Model Context Protocol，stdio，FastMCP） |
| 检索 | ChromaDB + 通义（Qwen）Embedding |
| 前端 | 原生 HTML / CSS / JavaScript，后端同源托管 |

## 架构

```
浏览器 (web/index.html)
        │  HTTP
        ▼
FastAPI (src/api/main.py)
        │
        ▼
BaseAgent (src/agent/qa_agent.py) ── OpenAI SDK ──► DeepSeek API
        │  MCP stdio
        ▼
MCP Server (src/mcp_server/mcp_server.py)
        │  campus_search 工具
        ▼
RAG (src/rag/rag.py) ──► ChromaDB (src/rag/chroma_db) + Qwen Embedding
        ▲
        │
   data/*.md 知识库
```

## 环境要求

- Windows / macOS / Linux
- Python **3.14**（在 3.14.4 上开发测试，`requirements.txt` 的版本锁定基于该版本）
- 可访问 DeepSeek 与通义（DashScope）API 的网络

## 快速开始

```bash
git clone https://github.com/ctwo22/SMU-QA.git
cd SMU-QA

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

复制环境变量模板，填入自己的 Key：

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

## 启动
```
使用vscode/Pycharm 配置好环境后 直接运行app.py ，点击运行终端的地址即可跳转
```
`或`
```bash
python app.py                 # 默认 http://127.0.0.1:8000
python app.py --port 8080     # 更换端口
python app.py --host 0.0.0.0  # 允许局域网访问
python app.py --reload        # 修改代码自动重启（会清空内存中的会话）
```

浏览器打开 <http://127.0.0.1:8000> 即可使用。

## 环境变量

| 变量 | 说明 | 示例 |
|---|---|---|
| `DEEPSEEK_MODEL` | 对话模型名 | `deepseek-flash` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | `sk-xxx` |
| `DEEPSEEK_BASE_URL` | DeepSeek 接口地址 | `https://api.deepseek.com` |
| `QWEN_MODEL` | 向量模型名 | `qwen3.7-text-embedding` |
| `QWEN_API_KEY` | 通义（DashScope）API Key | `sk-xxx` |
| `QWEN_BASE_URL` | 通义兼容模式地址 | `https://xxxxx` |

> 对话使用 DeepSeek、检索 Embedding 使用通义，**最少需要 两套 Key**。

## 目录结构

```
.
├── app.py                  # 统一启动入口
├── requirements.txt
├── .env.example            # 环境变量模板
├── data/                   # 校园知识库原文
│   ├── SYSTEM_Knowledge.md
│   ├── MyKnowledge.txt
│   └── About_SYSTEM.md
├── src/
│   ├── api/main.py         # FastAPI
│   ├── agent/qa_agent.py   # 对话 Agent
│   ├── agent/prompts.py    # 系统提示词
│   ├── log/logger.py       # 日志
│   ├── mcp_server/         # MCP Server（campus_search 工具）
│   └── rag/rag.py          # 向量检索与知识库构建
└── web/index.html          # 前端页面
```

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 提问，Body：`{"question": "..."}`（1–500 字） |
| GET | `/api/history` | 获取对话历史 |
| POST | `/api/reset` | 新建会话（清空服务端历史） |
| GET | `/api/health` | 健康检查（返回状态、模型名、工具列表） |

## 知识库更新

知识库原文位于 `data/` 目录。修改原文后重建向量库：  
`⚠注意：如果在data/目录下新增新的.txt/.md 文件，需要在rag.py 代码的第34行 添加新增文件名称`
```
运行src/rag/rag.py 文件导入新数据
```
`或`
```bash
python -m src.rag.rag
```

> 仓库中已包含构建好的 `src/rag/chroma_db/`，克隆后无需重建即可检索；
> 只有修改了 `data/` 下的文件才需要执行上面的重建命令（会调用 Embedding 接口，产生少量费用）。

## 日志

- 日志**日志写入文件，并在控制台输出 WARNING 及以上**：
  - 文件：项目根目录 `logs/`，首次运行自动创建
    - `logs/api.log`：FastAPI 服务日志
    - `logs/mcp_server.log`：MCP Server 日志
  - 控制台：只输出 `WARNING` 及以上级别，避免干扰 Uvicorn 的启动/访问日志
- 文件按大小轮转：单文件最大 **5MB**，保留 **3** 份备份（`api.log.1` ~ `api.log.3`），UTF-8 编码
- 级别由环境变量 `LOG_LEVEL` 控制，默认 `INFO`（例如 `LOG_LEVEL=DEBUG python app.py`）
- 格式：`时间 级别 进程号 logger名 | 消息`
- `logs/` 属于运行产物，已被 `.gitignore` 忽略
- 如果日志目录不可写，程序会自动降级为仅输出到 stderr，**不会中断服务启动**

## 常见问题


- **想更换对话模型？** 修改 `.env` 中的 `模型配置` 后重启，页面状态栏会同步显示新模型名。

## 免责声明

- 本项目为**个人学习与技术实践作品**，主要用于学习交流，不代表三明学院官方立场，与学校无隶属关系。
- 项目中的校园知识库内容整理自学校官网等**公开渠道**及作者个人补充，仅供参考；政策、收费、时间等信息可能随学校通知调整，**请以学校官方发布为准**。
- 本项目不采集、不上传任何个人隐私数据；对话记录仅保存在本地服务端内存中，重启服务即清空。
- 项目按“现状”提供，不对因使用本项目产生的任何直接或间接损失负责。
- 如项目内容涉及版权问题，请联系作者删除。

## License

本项目基于 [MIT License](LICENSE) 开源。