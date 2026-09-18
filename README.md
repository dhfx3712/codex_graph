# codex_graphrag

一个本地文章处理与语义检索项目：

1. 读取 Markdown 文章
2. 调用 LLM 生成文章摘要和知识图谱切片
3. 用本地 SentenceTransformer 模型生成切片向量
4. 写入 FAISS 索引和 SQLite 元数据/文本 store
5. 通过自然语言 query 检索切片并返回原始内容、摘要和结构化 JSON

## 目录结构

```text
.
├── articles/
│   ├── undo/                 # 待处理的 Markdown 文章
│   └── do/                   # 处理完成后归档的文章
├── results/
│   ├── articles.csv          # 文章摘要输出
│   ├── chunks.csv            # 切片、摘要和图谱输出
│   ├── chunks.faiss          # FAISS 向量索引
│   └── chunks_meta.sqlite3   # FAISS 映射 + chunk_texts 文本 store
├── summarize_pipeline.py     # 文章切片与 LLM 摘要流水线
├── build_chunk_faiss.py      # 构建 FAISS 索引和 SQLite store
├── query_chunk_faiss.py      # 语义查询脚本
└── requirements.txt          # Python 依赖
```

## 数据流

```text
articles/undo/*.md
        │
        ▼
summarize_pipeline.py
        │
        ├──► results/articles.csv
        └──► results/chunks.csv
                 │
                 ▼
        build_chunk_faiss.py
                 │
                 ├──► results/chunks.faiss
                 └──► results/chunks_meta.sqlite3
                          │
                          ▼
                 query_chunk_faiss.py
                          │
                          └──► 语义查询结果
```

## 环境准备

```bash
cd /home/ubuntu/codex_graphrag

# 如果还没有虚拟环境
python3 -m venv .venv

source .venv/bin/activate
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

主要依赖：

- `requests`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`

## 本地模型

默认使用本地缓存模型：

```text
~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```

FAISS 构建和查询脚本会默认设置：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

如果模型缓存不存在，需要先下载对应模型到该路径。

## 1. 文章摘要与切片处理

### 输入

把 Markdown 文件放进：

```text
articles/undo/
```

### 环境变量

优先读取 `/home/ubuntu/codex_english/.env`：

```bash
ARK_API_KEY=...
VOLC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
MODEL_NAME=doubao-seed-2.0-lite
SUMMARIZE_PIPELINE_WORKERS=4
```

也可以手动导出环境变量。

### 运行

```bash
source .venv/bin/activate
.venv/bin/python summarize_pipeline.py
```

处理完成后：

- Markdown 文件从 `articles/undo/` 移到 `articles/do/`
- 文章结果追加到 `results/articles.csv`
- 切片结果追加到 `results/chunks.csv`

## 2. 构建 FAISS 索引和文本 store

### Embedding 输入

从 `results/chunks.csv` 的 `summary_json` 中提取：

```text
summary_json.human_summary + "\n" + summary_json.core_argument
```

### 索引配置

- 模型：`sentence-transformers/all-MiniLM-L6-v2`
- 维度：`384`
- 归一化：`normalize_embeddings=True`
- 度量：`IndexFlatIP`，等价于余弦相似度
- 索引：`IndexIDMap2(IndexFlatIP(384))`
- 重建方式：全量重建

### 输出

```text
results/chunks.faiss
results/chunks_meta.sqlite3
```

SQLite 中核心表：

```sql
chunk_vectors(faiss_id, id, article_id, chunk_index);

chunk_texts(
    id, article_id, chunk_index,
    content, summary, summary_json,
    row_hash, created_at
);

build_info(key, value);
```

重复 `id` 的处理规则：

- 内容一致：忽略
- 内容不同：忽略新行，并写入 `WARNING` 日志

### 运行

```bash
source .venv/bin/activate

# 全量构建
.venv/bin/python build_chunk_faiss.py

# 只处理前 10 行，用于测试
.venv/bin/python build_chunk_faiss.py --max-rows 10
```

## 3. 语义查询

### 命令行查询

```bash
source .venv/bin/activate
.venv/bin/python query_chunk_faiss.py "美债目前的总规模是多少？" --top-k 5
```

### 返回结构

```json
{
  "query": "美债目前的总规模是多少？",
  "hits": [
    {
      "faiss_id": 67,
      "score": 0.5831988453865051,
      "id": "...",
      "article_id": "...",
      "chunk_index": 22,
      "content": "...",
      "summary": "...",
      "summary_json": {
        "human_summary": "...",
        "core_argument": "...",
        "entities": [],
        "relationships": [],
        "context_to_next": "...",
        "boundary_incomplete": false
      }
    }
  ]
}
```

### 程序化查询

```python
from query_chunk_faiss import search

hits = search("美债目前的总规模是多少？", top_k=5)

for hit in hits:
    print(hit["score"])
    print(hit["id"])
    print(hit["article_id"])
    print(hit["chunk_index"])
    print(hit["content"])
    print(hit["summary"])
    print(hit["summary_json"])
```

注意：

- `summary_json` 返回的是解析后的 Python `dict`
- 查询文本直接传自然语言字符串，不要传 JSON
- `search()` 每次会重新加载模型和索引，适合低频单次查询；高频批量查询建议进一步做模型/索引复用




• 已创建清理脚本 dedupe_undo_articles.py，并已执行。

  - 判断规则：按文件名比较 articles/undo 与 articles/do
  - 重复时提示“已处理”并删除对应 undo 文件
  - 本次结果：发现并删除 articles/undo/二战后电子信息.md
  - 其余 11 个 undo 文件未匹配，保留

  用法：

  python3 dedupe_undo_articles.py          # 实际删除
  python3 dedupe_undo_articles.py --dry-run  # 只查看，不删除




python serve_graph.py --host 0.0.0.0 --port 6134 --public-host



# 使用 AMD 的 DeepSeek 模型
codex --profile amd-deepseek

# 使用 AMD 的 Qwen 模型
codex --profile amd-qwen




• 重启命令给你（唯一的服务就是 serve_graph.py，静态页服务，端口 6134，README L259 确认的启动方式；web/index.html 只 fetch 同目录的 graph_data.json，无
  后端 API 依赖）：

  ## 一条搞定：杀旧 → 起新

  cd /home/ubuntu/codex_graphrag && pkill -f "serve_graph.py" ; sleep 1 && nohup .venv/bin/python serve_graph.py --host 0.0.0.0 --port 6134 --public-host > serve_graph.log 2>&1 & disown; sleep 2; ss -ltnp | grep 6134; tail -5 serve_graph.log

  预期最后输出类似：

  LISTEN  0  5  0.0.0.0:6134  ...  users:(("python",...))
  图网络页面已启动：http://0.0.0.0:6134/

  ## 想分两步执行

  # 1. 杀旧（无匹配进程时 pkill 返回 1，属正常）
  pkill -f "serve_graph.py"

  # 2. 起新
  cd /home/ubuntu/codex_graphrag
  nohup .venv/bin/python serve_graph.py --host 0.0.0.0 --port 6134 --public-host  10.0.0.14> serve_graph.log 2>&1 & disown

  ## 验证 / 应急

  # 确认只有一个实例在跑（重复起会因端口占用失败）
  pgrep -af serve_graph.py

  # 端口被别的进程占着时，先看清占用者再决定
  ss -ltnp | grep 6134

  # 确认页面可访问
  curl -I http://127.0.0.1:6134/
  curl -s -o /dev/null -w "graph_data.json: %{http_code} %{size_download}B\n" http://127.0.0.1:6134/graph_data.json

  需要浏览器访问外网地址时，加 --public-host <公网IP或域名>（仅用于日志提示，拒绝 TLS 请求时才用到）。

