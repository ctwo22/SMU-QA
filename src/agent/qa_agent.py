'''
Q&A agent 核心模块
'''

import asyncio
import json
import logging
import os
import sys
import time

from pathlib import Path
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI
from dotenv import load_dotenv
try:
    import prompts
except ModuleNotFoundError:  # 以 python -m src.agent.test 之类方式导入时
    from . import prompts
from contextlib import AsyncExitStack

# 显式指向项目根的 .env：不依赖启动时的工作目录（app.py 会 chdir，别的入口不会）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')

logger = logging.getLogger('agent')


# =======初始化Agent=========
class BaseAgent:
    # 引用类自动加载
    def __init__(self, model=os.getenv('DEEPSEEK_MODEL'), user_prompt=prompts.SYSTEM_USER_PROMPT,
                 react_prompt=prompts.SYSTEM_REACT_PROMPT):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            timeout=60.0,     # 单次请求上限（SDK 默认 600 秒，卡一次能卡十分钟）
            max_retries=1,    # 默认重试 2 次，叠加起来最坏能挂半小时
        )
        self.model = model
        # =====agent初始化全局记忆=======

        self.message = [  # <定义时就是字典啊>
            {'role': 'system', 'content': user_prompt},
            {'role': 'system', 'content': react_prompt},
        ]
        self.temperature = 0.5

        # 从mcp中获取函数调用解释
        self.tools_schema = []
        # 维持mcp会话
        self.mcp_session = None
        self.stdio_context = None
        self.exit_stack = AsyncExitStack()

    # 基本都是固定格式->拉取mcp进程
    async def init_mcp(self):
        """链接MCP Server 获取工具列表"""
        mcp_dir = Path(__file__).parent.parent / 'mcp_server' / 'mcp_server.py'
        server_params = StdioServerParameters(
            command=sys.executable, # 定义python解释器
            args=[str(mcp_dir)],    # 命令行参数 -C:\...\SMU_Q&A_Agent\.venv\Scripts\python.exe C:\...\src\mcp_server\mcp_server.py
            env=None,               # 是否读取.venv
        )
        read,write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        # mcp 的 ClientSession 参数名是 read_stream / write_stream（位置传参最稳）
        self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.mcp_session.initialize() # 握手初始化

        # 从mcp 拉取工具->OpenAI格式
        mcp_tools = await self.mcp_session.list_tools()
        for tool in mcp_tools.tools:
            self.tools_schema.append({
                "type":"function",
                "function":{
                    "name":tool.name,
                    "description":tool.description,
                    "parameters":tool.input_schema
                }
            })
        # print('成功加载MCP工具:', [t['function']['name'] for t in self.tools_schema])


    async def close_mcp(self):
        """关闭MCP会话，释放子进程"""
        await self.exit_stack.aclose()
        self.mcp_session = None
        self.tools_schema = []

    # 滑动窗口->大小=3
    def slide_memory(self):
        wid_size = sum(1 for msg in self.message if msg['role'] == 'user')
        if wid_size < 3:
            return self.message

        # 找到最近一次提问：它必须原样保留，只压缩它之前的历史
        last_user = max(i for i, msg in enumerate(self.message) if msg['role'] == 'user')
        history, current = self.message[:last_user], self.message[last_user:]

        sum_raw = self.sum_memory(history)
        self.message = [
            self.message[0],
            self.message[1],
            {'role': 'system', 'content': f'【历史摘要】{sum_raw}'},
            *current,
        ]
        return self.message

    # 摘要记忆
    def sum_memory(self, message):
        # 只喂纯文本对话：带 tool_calls 的 assistant / tool 消息顺序敏感，回传给API会400
        history = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in message[2:]
            if msg['role'] in ('user', 'assistant') and msg.get('content')
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                *history,
                {'role': 'user', 'content': '帮我对这些历史对话进行总结'}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ''

    # =========这里就开始调用LLM了，也就是说-这里的messages就是最终给LLM的记忆============
    def get_completions(self):

        # 有工具才回去判断是否需要调用工具，如果没工具直接空
        kwargs = {}
        if self.tools_schema:
            kwargs['tools'] = self.tools_schema        # 不传 tools，模型永远不会发起工具调用
            kwargs['tool_choice'] = 'auto'

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.slide_memory(),
            **kwargs,
            temperature=self.temperature,
        )
        response_json = response.choices[0].message.model_dump()
        # (筛选)只保留 OpenAI 认可的字段，避免把 reasoning_content / annotations 回传报错
        response_json = {
            'role': response_json.get('role') or 'assistant',
            'content': response_json.get('content'),
            'tool_calls': response_json.get('tool_calls'),
        }
        self.message.append(response_json)  # ai回答默认'role':'assistant'
        return response_json

    async def _run_tool_call(self, tool_call: dict) -> None:
        """执行一次工具调用，并把结果写回历史。

        不管成功、失败还是参数坏了，都必须追加一条 role=tool 的消息：
        assistant 上每个 tool_calls.id 都要有回应，否则下一次请求必然 400，
        而且这个坏状态会一直留在内存里，直到重启服务或点“新建会话”。
        """
        call_id = tool_call.get('id')
        function = tool_call.get('function') or {}
        function_name = function.get('name') or 'unknown'
        raw_args = function.get('arguments') or '{}'

        # 模型偶尔会吐坏 JSON（被截断、用单引号、给空串），解析失败也要回应这个 id
        try:
            function_args = json.loads(raw_args)
            if not isinstance(function_args, dict):
                raise ValueError('arguments 不是 JSON 对象')
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning('工具参数解析失败 tool=%s raw=%r err=%s', function_name, raw_args, exc)
            self.message.append({
                'role': 'tool',
                'tool_call_id': call_id,
                'name': function_name,
                'content': '[参数解析失败] arguments 必须是合法 JSON 对象，请重新调用工具。',
            })
            return

        logger.info('调用工具 tool=%s args=%s', function_name, list(function_args))

        try:
            mcp_result = await self.mcp_session.call_tool(function_name, arguments=function_args)
            tool_content = '\n'.join(c.text for c in mcp_result.content if c.type == 'text')
            if getattr(mcp_result, 'isError', False):
                logger.warning('工具返回错误 tool=%s content=%r', function_name, tool_content[:200])
                tool_content = f'[工具报错] {tool_content or "未知错误"}'
            else:
                logger.info('工具调用完成 tool=%s result_chars=%d', function_name, len(tool_content))
            tool_content = tool_content or '(工具没有返回文本内容)'
        except Exception:
            # 关键：不 raise，把失败当成一次“没有结果的工具调用”喂回模型，历史保持合法
            logger.exception('工具调用失败 tool=%s', function_name)
            tool_content = '[工具调用失败] 本轮检索没有拿到结果，请如实告知用户稍后再试，不要编造答案。'

        self.message.append({
            'role': 'tool',
            'tool_call_id': call_id,
            'name': function_name,
            'content': tool_content,
        })

    # ========这里是最后得到的了 这个query已经通过拼接了========
    async def chat(self, user_query: str, max_turn=5, budget_seconds=120):
        """跑一轮完整问答。

        - get_completions 是同步阻塞调用，丢到线程里跑，避免把事件循环卡死
          （否则生成期间刷新页面、健康检查、Ctrl+C 全都动不了）；
        - budget_seconds 是整轮对话的总预算，超了直接中止，不让用户干等；
        - 工具失败也一定写回 tool 消息，见 _run_tool_call。
        """
        self.message.append({'role': 'user', 'content': user_query})
        started = time.monotonic()

        for _ in range(max_turn):
            if time.monotonic() - started > budget_seconds:
                logger.warning('本轮对话超时 budget=%ss，中止', budget_seconds)
                return '（这次回答耗时过长，已中止。可以把问题拆小一点再问一次。）'

            message = await asyncio.to_thread(self.get_completions)

            tool_calls = message.get('tool_calls')
            if tool_calls:
                for tool_call in tool_calls:
                    await self._run_tool_call(tool_call)
                continue  # 带着工具结果再问一次模型

            if message.get('content'):
                print(f"A: {message['content']}")
                return message['content']

        print('A: 达到最大轮次仍未得到最终回答')
        return None
