"""
基于 Streamlit 的 RAG 问答 Web 应用
使用 LlamaIndex 框架和本地 HuggingFace 模型进行向量化
支持 DeepSeek API 和 Mock 层，配置流式输出
"""

import os
import logging
from typing import List
from pathlib import Path
from streamlit.runtime.scriptrunner import RerunData, RerunException

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from config import Config
from mock_llm import MockDeepSeek

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="RAG 问答系统",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e0e6df 0%, #a9d4c9 100%);
    }
    .main-header {
        text-align: center;
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .user-message {
        background-color: #e0e7ff;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin-bottom: 20px;
        margin-left: auto;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #f3f4f6;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin-bottom: 20px;
        margin-right: auto;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .avatar {
        font-size: 2rem;
        margin-right: 10px;
    }
    .info-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_embed_model():
    """初始化嵌入模型（缓存）"""
    try:
        logger.info("正在加载嵌入模型...")
        embed_model = HuggingFaceEmbedding(
            model_name=Config.MODEL_PATH,
            device="cpu",
            embed_batch_size=32
        )
        logger.info("嵌入模型加载完成")
        return embed_model
    except Exception as e:
        logger.error(f"加载嵌入模型失败: {e}")
        st.error(f"加载嵌入模型失败: {e}")
        return None


@st.cache_resource
def initialize_llm(api_key: str, use_mock: bool = False):
    """初始化LLM（缓存）"""
    try:
        if use_mock or Config.is_mock():
            logger.info("使用模拟DeepSeek LLM")
            return MockDeepSeek(
                model=Config.DEEPSEEK_MODEL,
                api_key="mock_key",
                temperature=Config.DEEPSEEK_TEMPERATURE
            )
        else:
            if not api_key:
                raise ValueError("API密钥不能为空")
            
            logger.info("使用真实DeepSeek API")
            return DeepSeek(
                model=Config.DEEPSEEK_MODEL,
                api_key=api_key,
                temperature=Config.DEEPSEEK_TEMPERATURE
            )
    except Exception as e:
        logger.error(f"初始化LLM失败: {e}")
        st.error(f"初始化LLM失败: {e}")
        return None


def load_documents(data_path: str) -> List:
    """加载文档"""
    try:
        logger.info(f"正在加载文档，路径: {data_path}")
        
        # 检查目录是否存在
        if not os.path.exists(data_path):
            logger.warning(f"目录不存在: {data_path}")
            return []
        
        # 读取所有.txt和.md文件
        documents = SimpleDirectoryReader(
            data_path,
            required_exts=[".txt", ".md"],
            recursive=True
        ).load_data()
        
        logger.info(f"加载了 {len(documents)} 个文档")
        return documents
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        st.error(f"加载文档失败: {e}")
        return []


def create_index(documents, llm, embed_model):
    """创建向量索引"""
    try:
        if not documents:
            logger.warning("没有文档可索引")
            return None
        
        logger.info("正在创建向量索引...")
        
        # 配置LlamaIndex
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # 创建索引
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        logger.info("向量索引创建完成")
        return index
    except Exception as e:
        logger.error(f"创建索引失败: {e}")
        st.error(f"创建索引失败: {e}")
        return None


def render_sidebar():
    """渲染侧边栏（只创建一次控件）"""
    with st.sidebar:
        st.title("🤖 专家助手")
        st.markdown("---")
        
        # 简介信息
        st.subheader("📖 简介")
        st.info("""
        **NK助手** 是一个基于 RAG（检索增强生成）技术的智能问答系统。
        
        主要功能：
        - 📚 自动加载知识库文档
        - 🔍 智能检索相关内容
        - 💬 基于上下文的精准回答
        - ⚡ 实时流式输出
        """)
        
        st.markdown("---")
        
        # API Key 输入（使用固定的key确保只创建一次）
        st.subheader("🔑 配置")
        
        # 环境选择
        use_mock = st.checkbox("使用模拟层 (Mock)", value=False, key="use_mock_main",
                           help="勾选后使用模拟API，无需真实API密钥")
        
        if not use_mock:
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                key="api_key_main",
                placeholder="请输入您的 DeepSeek API 密钥",
                help="在 https://platform.deepseek.com/ 获取"
            )
        else:
            api_key = "mock_key"
            st.success("✅ 已启用模拟层模式")
        
        # 保存配置到session state
        st.session_state.use_mock = use_mock
        st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # 重新加载按钮
        if st.button("🔄 重新加载知识库", use_container_width=True, key="reload_btn_main"):
            st.session_state.reload_index = True
            # 使用rerun确保状态更新
            st.rerun()
        
        # 系统状态（实时更新）
        st.subheader("📊 系统状态")
        
        if "document_count" in st.session_state and st.session_state.reload_index == True:
            documents = load_documents(Config.DATA_PATH)
            st.session_state.document_count = len(documents)
            st.metric("文档数量", st.session_state.document_count)
        # else:
        #     st.metric("文档数量", 0)
        
        if "index_status" in st.session_state:
            if st.session_state.index_status == "初始化":
                pass
            else:
                status_color = "🟢" if st.session_state.index_status == "就绪" else "🔴"
                st.write(f"{status_color} 索引状态: {st.session_state.index_status}")
        else:
            st.write("🔴 索引状态: 未就绪")
        
        # 显示加载进度
        if "loading_progress" in st.session_state and st.session_state.loading_progress:
            st.info("🔄 正在加载知识库...")
        
        st.markdown("---")
        st.caption(f"当前环境: {Config.get_env_name()}")


def render_chat_history():
    """渲染聊天历史"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 聊天容器
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                    <div class="user-message">
                        <div class="avatar">👤</div>
                        <div>{message["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                    <div class="assistant-message">
                        <div class="avatar">🤖</div>
                        <div>{message["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_chat_input(index, llm):
    """渲染聊天输入框"""
    # 使用 st.chat_input 替代底部固定输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        st.chat_message("user").write(prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    if index is None:
                        response_text = "抱歉，知识库未初始化，请先重新加载知识库。"
                    else:
                        # 创建查询引擎
                        query_engine = index.as_query_engine(
                            llm=llm,
                            similarity_top_k=3,
                            streaming=False
                        )
                        
                        # 查询
                        response = query_engine.query(prompt)
                        response_text = str(response)
                    
                    # 显示回答
                    st.write(response_text)
                    
                    # 添加到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                except Exception as e:
                    error_msg = f"生成回答时出错: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)


def main():
    """主函数"""
    # 初始化session state
    if "reload_index" not in st.session_state:
        st.session_state.reload_index = False
    
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    
    if "index_status" not in st.session_state:
        st.session_state.index_status = "初始化"
    
    if "use_mock" not in st.session_state:
        st.session_state.use_mock = False
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "index" not in st.session_state:
        st.session_state.index = None
    
    # 第一步：渲染侧边栏（获取配置）
    render_sidebar()
    
    # 第二步：获取配置（从侧边栏的session state中读取）
    use_mock = st.session_state.use_mock
    api_key = st.session_state.api_key
    
    # 第三步：初始化模型（仅当需要时）
    if st.session_state.embed_model is None or st.session_state.reload_index:
        with st.spinner("正在初始化模型..."):
            embed_model = initialize_embed_model()
            st.session_state.embed_model = embed_model
            st.session_state.model_initialized = embed_model is not None
    
    # 第四步：初始化LLM（仅当需要时）
    if st.session_state.llm is None or st.session_state.reload_index:
        llm = initialize_llm(api_key, use_mock)
        st.session_state.llm = llm
    
    # 第五步：加载文档和创建索引（当需要重新加载或首次加载时）
    if st.session_state.index is None or st.session_state.reload_index:
        if st.session_state.embed_model and st.session_state.llm:
            try:
                # 显示加载进度
                st.session_state.loading_progress = True
                
                with st.spinner("正在加载知识库..."):
                    documents = load_documents(Config.DATA_PATH)
                    
                    # 实时更新文档数量
                    st.session_state.document_count = len(documents)
                    
                    if documents:
                        index = create_index(documents, st.session_state.llm, st.session_state.embed_model)
                        st.session_state.index = index
                        st.session_state.index_status = "就绪"
                        # st.success(f"✅ 知识库加载完成！共 {len(documents)} 个文档")
                    else:
                        st.session_state.index = None
                        st.session_state.index_status = "无文档"
                        st.warning("⚠️ 未找到文档，请在 data/ 目录下添加 .txt 或 .md 文件")
                    
                    st.session_state.reload_index = False
                    st.session_state.loading_progress = False
                    
            except Exception as e:
                st.session_state.index_status = "加载失败"
                st.error(f"❌ 加载知识库失败: {str(e)}")
                logger.error(f"加载知识库失败: {e}")
                st.session_state.loading_progress = False
        else:
            if not st.session_state.model_initialized:
                st.error("模型初始化失败，无法加载知识库")
            st.session_state.index_status = "未就绪"
    
    # 第六步：渲染主界面内容
    # 主界面标题
    st.markdown('<h1 class="main-header">资料查询小助手</h1>', unsafe_allow_html=True)
    
    # 显示系统提示
    if st.session_state.index_status == "就绪":
        st.info(f"📊 知识库已就绪，包含 {st.session_state.document_count} 个文档")
    elif st.session_state.index_status == "无文档":
        st.warning("⚠️ 请在 data/ 目录下添加文档并点击'重新加载知识库'")
    elif st.session_state.index_status == "未就绪":
        st.info("ℹ️ 点击侧边栏的'重新加载知识库'按钮开始")
    
    # 第七步：渲染聊天历史
    render_chat_history()
    
    # 第八步：渲染聊天输入
    render_chat_input(st.session_state.get("index"), st.session_state.get("llm"))


if __name__ == "__main__":
    main()
