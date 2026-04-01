import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from config import Config

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入模拟层
if Config.is_mock():
    from mock_llm import MockDeepSeek

logger.info(f"=== 启动应用 - 环境: {Config.get_env_name()} ===")

# 配置路径
DATA_PATH = Config.DATA_PATH
MODEL_PATH = Config.MODEL_PATH

# 根据环境选择LLM
if Config.is_mock():
    # 使用模拟层（测试环境）
    logger.info("使用模拟DeepSeek LLM")
    llm = MockDeepSeek(
        model=Config.DEEPSEEK_MODEL,
        api_key="mock_key",
        temperature=Config.DEEPSEEK_TEMPERATURE
    )
else:
    # 使用真实DeepSeek API（生产环境）
    api_key = Config.DEEPSEEK_API_KEY
    if not api_key:
        logger.error("DEEPSEEK_API_KEY 环境变量未设置")
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    
    logger.info("使用真实DeepSeek API")
    llm = DeepSeek(
        model=Config.DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=Config.DEEPSEEK_TEMPERATURE
    )

# 初始化嵌入模型
logger.info("正在加载嵌入模型(首次运行会比较慢)...")
embed_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH,
    device="cpu",  # 使用CPU,如果GPU可用可以改为"cuda"
    embed_batch_size=32  # 批量处理大小
)
logger.info("嵌入模型加载完成")

# 加载文档
logger.info(f"正在加载文档，路径: {DATA_PATH}")
documents = SimpleDirectoryReader(DATA_PATH).load_data()
logger.info(f"加载了 {len(documents)} 个文档")

# 创建向量索引
logger.info("正在创建向量索引并生成嵌入向量(这可能需要几分钟时间)...")
index = VectorStoreIndex.from_documents(
    documents=documents,
    llm=llm,
    embed_model=embed_model,
    show_progress=True  # 显示进度
)
logger.info("向量索引创建完成")

# 创建查询引擎
query_engine = index.as_query_engine(llm=llm)
logger.info("查询引擎创建完成")

# 测试查询
test_question = "这个文档的主要内容是什么？"
logger.info(f"测试问题: {test_question}")
try:
    response = query_engine.query(test_question)
    logger.info("回答生成完成")
    print("\n回答:")
    print(response.response)
    print("\n" + "=" * 50 + "\n")
except Exception as e:
    logger.error(f"测试过程中出错: {e}")
    print("测试过程中出错：", e)
    import traceback
    traceback.print_exc()

# 主循环
logger.info("进入交互模式")
print("\n请输入您的问题(输入'quit'结束):")
while True:
    try:
        question = input("问题：")
    except EOFError:
        logger.info("检测到EOF，退出程序")
        break
    
    if question.lower() == "quit":
        logger.info("用户请求退出")
        break
    
    try:
        logger.info(f"处理问题: {question[:50]}...")
        response = query_engine.query(question)
        print("\n回答:")
        print(response.response)
        print("\n" + "-" * 50 + "\n")
    except Exception as e:
        logger.error(f"查询过程中出错: {e}")
        print("查询过程中出错：", e)
