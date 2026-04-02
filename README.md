# 垂直领域专家助手 - RAG 问答系统

基于 LlamaIndex 框架的智能问答系统，支持本地向量检索和云端大语言模型。

## 项目特性

- 🚀 **双模式运行**: 支持 Mock 模拟层和真实 DeepSeek API
- 📚 **自动知识库管理**: 自动加载和索引文档
- 🔍 **智能检索**: Top-K=3 相似度检索
- 💬 **流式对话**: 实时流式输出支持
- 🎨 **现代化界面**: 基于 Streamlit 的美观 Web 界面
- ⚡ **高性能**: 本地向量化，快速响应

## 项目结构

```
nk_assistant/
├── app.py                    # 命令行版本的 RAG 应用
├── web_app.py               # Streamlit Web 应用
├── config.py                # 配置管理
├── mock_llm.py              # DeepSeek API 模拟层
├── test_mock_llm.py         # 模拟层测试
├── requirements.txt         # 核心依赖
├── requirements_web.txt     # Web 应用依赖
├── data/                   # 知识库目录
│   ├── test.txt
│   ├── 技术文档.md
│   └── 使用说明.txt
└── model_cache/            # 嵌入模型缓存
```

## 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# Web 应用依赖（可选）
pip install -r requirements_web.txt
```

### 2. 运行方式

#### 命令行版本
```bash
# 使用模拟层（测试）
python app.py

# 使用真实 API
export DEEPSEEK_API_KEY="your_api_key"
APP_ENV=production python app.py
```

#### Web 版本（推荐）
```bash
# 启动 Web 应用
streamlit run web_app.py

# 或使用启动脚本
./run_web.sh
```

然后在浏览器打开：`http://localhost:8501`

## 环境配置

### 环境变量

```bash
# 应用环境（mock/production/test）
export APP_ENV=mock

# DeepSeek API Key（生产环境必需）
export DEEPSEEK_API_KEY="your_api_key"

# 日志级别（DEBUG/INFO/WARNING/ERROR）
export LOG_LEVEL=INFO
```

### 环境说明

- **mock**: 使用模拟层，无需 API 密钥，适合测试
- **production**: 使用真实 DeepSeek API，需要 API 密钥
- **test**: 测试环境配置

## Web 应用功能

### 侧边栏
- 📖 **简介**: 系统功能介绍
- 🔑 **配置**: API 密钥输入和模拟层开关
- 🔄 **重新加载知识库**: 重新加载文档
- 📊 **系统状态**: 显示文档数量和索引状态

### 主界面
- 顶部居中显示"垂直领域专家助手"标题
- 聊天历史记录区域（区分用户和助手消息）
- 底部固定聊天输入框

## 技术栈

- **LlamaIndex**: 向量检索和查询引擎
- **HuggingFace**: BAAI/bge-small-zh-v1.5 嵌入模型
- **DeepSeek API**: 大语言模型支持
- **Streamlit**: Web 框架
- **Sentence-Transformers**: 文本向量化

## 核心功能实现

### 1. 文档加载
```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    data_path,
    required_exts=[".txt", ".md"],
    recursive=True
).load_data()
```

### 2. 向量索引
```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    show_progress=True
)
```

### 3. 智能检索
```python
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    streaming=True
)

response = query_engine.query(question)
```

## Mock 模拟层

模拟层提供了完整的 DeepSeek API 接口，支持：

- ✅ 智能响应生成
- ✅ 多种错误场景模拟
- ✅ 调用统计功能
- ✅ 与真实 API 完全兼容的接口

测试模拟层：
```bash
python test_mock_llm.py
```

## 知识库管理

### 添加文档
1. 将 `.txt` 或 `.md` 文件放入 `data/` 目录
2. 点击侧边栏的"重新加载知识库"按钮
3. 等待索引创建完成

### 支持的格式
- 纯文本文件 (`.txt`)
- Markdown 文件 (`.md`)

### 推荐做法
- 将大文档拆分成小文件（< 10MB）
- 使用清晰的标题结构
- 添加适当的格式化

## 性能优化

### 加速向量检索
- 减少 Top-K 值
- 使用更小的嵌入模型
- 增加相似度阈值

### 减少内存使用
- 定期清理未使用的文档
- 使用量化模型
- 启用分块处理

### GPU 加速
```python
embed_model = HuggingFaceEmbedding(
    model_name=model_path,
    device="cuda"  # 使用 GPU
)
```

## 常见问题

### Q: 如何获取 DeepSeek API Key？
A: 访问 https://platform.deepseek.com/ 注册并获取 API Key。

### Q: 首次运行很慢？
A: 需要下载嵌入模型，约 500MB，请耐心等待。

### Q: 如何提高回答质量？
A: 提供更准确的知识库文档，提问时明确具体。

### Q: 支持多语言吗？
A: 主要支持中文，使用专门的中文嵌入模型。

## 开发指南

### 添加新的响应类型
编辑 `mock_llm.py` 中的 `_generate_mock_response` 方法。

### 自定义界面样式
编辑 `web_app.py` 中的 CSS 样式部分。

### 修改检索策略
调整 `similarity_top_k` 参数和相关配置。

## 部署

### Streamlit Cloud
1. 将代码推送到 GitHub
2. 在 Streamlit Cloud 中导入仓库
3. 配置环境变量
4. 自动部署

### Docker 部署
```bash
docker build -t rag-assistant .
docker run -p 8501:8501 rag-assistant
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

享受您的智能问答体验！🚀
