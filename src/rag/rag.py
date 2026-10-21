import collections
import logging
import os
from pathlib import Path

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from mdclense.parser import MarkdownParser # 洗.md文件

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = Path(__file__).parent.parent.parent.absolute()

# ====初始化本地持久客户端======
# ⚠用绝对路径：否则 ./chroma_db 跟着 cwd 跑，同一个库会在不同目录各建一份⚠
client = chromadb.PersistentClient(path=str(Path(__file__).parent / 'chroma_db'))

load_dotenv(base_dir / '.env')  # 显式指定，不再依赖 cwd 向上查找 -⚠ 直接找当前项目的.venv ⚠
if not os.getenv('QWEN_API_KEY'):
    raise RuntimeError(f'没读到 QWEN_API_KEY，请检查 {base_dir / ".env"}')

embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('QWEN_API_KEY'),
    api_base = os.getenv('QWEN_BASE_URL'),
    model_name=os.getenv('QWEN_MODEL'),
)

collection = client.get_or_create_collection(
    name='Info_QA',
    embedding_function = embedding_func,
)

file_list = [
    base_dir / 'data' / 'SYSTEM_Knowledge.md',
    base_dir / 'data' / 'MyKnowledge.txt',
    base_dir / 'data' / 'About_SYSTEM.md',
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n\n','\n','。','！','？','，',''],
)

def build_index(force: bool = False):
    '''
    把 data 下的知识文件切块写进向量库。
    库非空就直接跳过（不再每次 import 都重新 embedding 一遍）；
    force=True 用于改了 data 里的文件后重建。
    '''
    if not force and collection.count() > 0:
        return collection.count()

    if force:
        existed = collection.get()['ids']
        if existed:
            collection.delete(ids=existed)

    all_chunks = []
    all_metadata = []
    all_ids = []

    for file_path in file_list:
        if not file_path.exists():
            print(f'File {file_path} does not exist')
            continue

        with open(file_path,encoding='utf-8') as f:
            text = f.read()

        # 如果文件为'.md'则洗文件 ->.suffix(取扩展名)
        if file_path.suffix.lower() == '.md':
            text = MarkdownParser().parse(text)

        chunks = text_splitter.split_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_metadata.append({
                'source': file_path.name,
                'chunk_index': index,
            })

            # ids唯一 ,防止撞名
            all_ids.append(f'{file_path.name}_{index}')

    if len(all_chunks) > 0:
        batch_size=10
        for i in range(0,len(all_chunks),batch_size):
            batch_docs = all_chunks[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            batch_metas = all_metadata[i:i+batch_size]

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )

    return collection.count()


if collection.count() == 0:
    build_index()

# 设定最大相关值
MIN_DISTANCE = float(1.15)

# 检索不到相关内容时的返回值。给模型一个明确的“没有资料”信号，
NO_RELEVANT_DOC = '[无相关资料] 知识库里没有检索到与这个问题相关的内容。'

logger = logging.getLogger('mcp.server.rag')


def Knowledges(query: str, Top_n: int = 2):
    '''
    检索官方和作者自写的校园知识库，返回拼接的格式化文本。

    距离超过 MIN_DISTANCE 的片段会被丢弃；一条都不剩时返回明确的“无相关资料”标记，
    让模型走“找不到答案”的分支，而不是拿不相关的内容编答案。
    '''
    results = collection.query(
        query_texts=[query],
        n_results=Top_n,
        include=['documents', 'metadatas', 'distances'],
    )
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = (results.get('distances') or [[]])[0]

    if not docs:
        logger.info('tool=rag 没检索到任何片段 query_len=%d', len(query))
        return NO_RELEVANT_DOC

    format_docs = []
    for i, doc in enumerate(docs):
        source = metadatas[i].get('source', '未知')
        distance = distances[i] if i < len(distances) else None

        if distance is not None and distance > MIN_DISTANCE:
            logger.info('tool=rag 丢弃不相关片段 source=%s distance=%.3f', source, distance)
            continue

        logger.info('tool=rag 命中 source=%s distance=%s', source,
                    f'{distance:.3f}' if distance is not None else '未知')
        format_docs.append(f'【来源: {source}】\n{doc}')

    if not format_docs:
        return NO_RELEVANT_DOC
    return '\n\n-----\n\n'.join(format_docs)

# 重新更新数据库 -如果有新文件的加入，需要添加进file_list
if __name__ == '__main__':
    build_index(True)
    print('重新导入完成')