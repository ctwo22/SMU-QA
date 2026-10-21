import os
import sys
import time
from pathlib import Path


from openai import OpenAI, base_url

# 以脚本方式被 stdio 拉起时，sys.path[0] 是 src/mcp_server，项目根不在路径里
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmcp import FastMCP
from src.log.logger import setup_logging
from src.rag.rag import Knowledges, client

logger = setup_logging('mcp.server', 'mcp_server.log')

mcp = FastMCP('SMU_Q&A')


@mcp.tool()
def campus_search(query: str):
    """
    搜索三明学院内部的规章制度、新生指南、常见问答等校园专属知识库。
    当用户询问任何有关于三明学院的知识的时候，请优先调用此工具。
    """
    started = time.perf_counter()
    try:
        result = Knowledges(query)
    except Exception:
        logger.exception('tool=campus_search 失败 q_len=%d', len(query))
        raise

    logger.info('tool=campus_search ok q_len=%d result_chars=%d cost=%.0fms',
                len(query), len(result or ''),
                (time.perf_counter() - started) * 1000)
    return result



if __name__ == '__main__':
    logger.info('MCP server 启动 transport=stdio')
    try:
        mcp.run(transport='stdio')
    finally:
        logger.info('MCP server 退出')