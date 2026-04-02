# 快速启动指南

## 5分钟开始使用

### 1. 启动 Web 应用

```bash
# 最简单的方式
streamlit run web_app.py
```

### 2. 访问界面

浏览器打开：http://localhost:8501

### 3. 配置并使用

#### 使用模拟层（快速测试）
1. ✅ 勾选侧边栏的"使用模拟层"
2. 🔄 点击"重新加载知识库"
3. 💬 在底部输入框输入问题

#### 使用真实 API
1. 🔑 在 https://platform.deepseek.com/ 获取 API Key
2. 🔑 在侧边栏输入 API Key
3. 🔄 点击"重新加载知识库"
4. 💬 开始提问

## 功能演示

### 示例问题

```
这个系统的主要功能是什么？
如何添加新的文档？
遇到错误怎么办？
```

### 侧边栏操作

- **📖 简介** - 查看系统功能
- **🔑 配置** - 设置 API 或切换模式
- **🔄 重新加载** - 更新知识库
- **📊 状态** - 查看系统信息

## 知识库管理

### 添加文档

```bash
# 将文档放入 data/ 目录
cp your_document.txt data/

# 或创建新文档
echo "你的知识内容" > data/新文档.txt
```

然后点击"重新加载知识库"按钮。

### 支持的格式

- `.txt` - 纯文本文件
- `.md` - Markdown 文件

## 测试模拟层

```bash
# 运行测试
python test_mock_llm.py

# 测试 Web 应用功能
python test_web_app.py
```

## 常用命令

```bash
# 启动 Web 应用
streamlit run web_app.py

# 命令行版本
python app.py

# 测试
python test_mock_llm.py
python test_web_app.py
```

## 故障排除

### 问题：端口被占用
```bash
# 使用其他端口
streamlit run web_app.py --server.port 8502
```

### 问题：模型加载慢
- 首次加载需要下载模型，请耐心等待
- 模型约 500MB，下载时间取决于网络速度

### 问题：无响应
1. 检查是否使用了模拟层
2. 确认文档已正确加载
3. 查看浏览器控制台错误信息

## 性能优化

### 加速启动
```python
# 编辑 web_app.py
embed_model = HuggingFaceEmbedding(
    model_name=Config.MODEL_PATH,
    device="cuda"  # 如果有GPU
)
```

### 减少内存
- 减少 data/ 目录中的文档数量
- 降低 Top-K 值（默认为 3）

## 下一步

- 📖 阅读完整的 [README.md](README.md)
- 🎨 自定义界面样式
- 🔧 调整检索参数
- 📊 监控系统性能

---

需要帮助？查看 [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) 获取详细说明。
