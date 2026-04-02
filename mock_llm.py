"""
模拟DeepSeek LLM - 用于测试和开发
模拟真实API的各种响应情况，包括成功和错误处理
"""

import logging
from typing import List, Any, Optional
from pydantic import PrivateAttr
from llama_index.core.llms import ChatMessage, CompletionResponse, LLM
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.base.llms.types import ChatResponse, LLMMetadata

logger = logging.getLogger(__name__)


class MockDeepSeek(LLM):
    """
    模拟DeepSeek LLM类
    接口与真实DeepSeek保持一致，便于后续替换
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            **kwargs
        )
        self._api_key = api_key
        self._call_count = 0
        
        logger.info(f"[MOCK] 初始化模拟DeepSeek LLM - 模型: {model}, 温度: {temperature}")
    
    @property
    def api_key(self):
        """API密钥属性"""
        return self._api_key
    
    def _get_model_name(self):
        """获取模型名称"""
        return self.__dict__.get('model', 'deepseek-chat')
    
    def _get_temperature(self):
        """获取温度参数"""
        return self.__dict__.get('temperature', 0.1)
    
    @property
    def metadata(self):
        """返回模型元数据"""
        return LLMMetadata(
            context_window=4096,
            num_output=2048,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self._get_model_name()
        )
    
    @llm_chat_callback()
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """
        模拟chat方法
        与真实DeepSeek接口一致
        """
        self._call_count += 1
        logger.info(f"[MOCK] 聊天生成 - 消息数: {len(messages)}, 调用次数: {self._call_count}")
        
        # 提取用户消息内容
        user_content = ""
        for msg in messages:
            if msg.role == "user":
                # 确保content是字符串类型
                content_str = str(msg.content)
                user_content += content_str + "\n"
        
        response_text = self._generate_mock_response(user_content, "chat")
        
        # 返回ChatResponse对象
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text),
            finish_reason="stop"
        )
    
    def predict(self, prompt: str, **kwargs) -> str:
        """
        模拟predict方法
        与真实DeepSeek接口一致
        """
        self._call_count += 1
        logger.info(f"[MOCK] 预测生成 - 调用次数: {self._call_count}")
        
        # 确保prompt是字符串类型
        prompt_str = str(prompt)
        response_text = self._generate_mock_response(prompt_str, "predict")
        return response_text
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        模拟complete方法
        与真实DeepSeek接口一致
        """
        self._call_count += 1
        logger.info(f"[MOCK] 完整文本生成 - 调用次数: {self._call_count}")
        
        # 确保prompt是字符串类型
        prompt_str = str(prompt)
        response_text = self._generate_mock_response(prompt_str, "complete")
        return CompletionResponse(text=response_text)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        """模拟流式聊天（简化实现）"""
        response = self.chat(messages, **kwargs)
        yield response
    
    def stream_complete(self, prompt: str, **kwargs):
        """模拟流式补全（简化实现）"""
        response = self.complete(prompt, **kwargs)
        yield response
    
    async def achat(self, messages: List[ChatMessage], **kwargs):
        """异步聊天（简化实现）"""
        return self.chat(messages, **kwargs)
    
    async def acomplete(self, prompt: str, **kwargs):
        """异步补全（简化实现）"""
        return self.complete(prompt, **kwargs)
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs):
        """异步流式聊天（简化实现）"""
        async for response in self.stream_chat(messages, **kwargs):
            yield response
    
    async def astream_complete(self, prompt: str, **kwargs):
        """异步流式补全（简化实现）"""
        async for response in self.stream_complete(prompt, **kwargs):
            yield response
    
    def _generate_mock_response(self, prompt: str, mode: str) -> str:
        """
        生成模拟响应
        根据输入内容返回智能化的预设响应
        """
        # 确保prompt是字符串类型
        prompt_str = str(prompt)
        prompt_lower = prompt_str.lower()

        # 测试文档主要内容
        if "主要内容" in prompt_str or "summary" in prompt_lower or "总结" in prompt_str:
            return """根据文档内容，这是一个关于技术文档的示例。文档包含了多个章节，涵盖了基础知识、实践应用和高级特性。

主要内容包括：
1. 系统架构和设计原理
2. 核心功能模块说明
3. API接口文档
4. 最佳实践和注意事项
5. 常见问题解答

该文档旨在帮助开发者快速理解和使用相关技术栈，提供了从入门到精通的完整指南。"""

        # 测试代码相关问题
        elif "代码" in prompt_str or "code" in prompt_lower or "函数" in prompt_str:
            return """关于代码相关的提问，这里提供一个示例响应：

```python
def example_function(data):
    '''
    示例函数
    '''
    result = process_data(data)
    return result
```

这段代码展示了基本的数据处理流程，具体实现需要根据实际需求进行调整。"""

        # 测试错误处理
        elif "错误" in prompt_str or "error" in prompt_lower or "异常" in prompt_str:
            return """关于错误处理，建议采用以下最佳实践：

1. 使用try-except捕获异常
2. 记录详细的错误日志
3. 提供友好的错误提示
4. 实现重试机制
5. 设置超时和熔断保护

这样可以提高系统的健壮性和可维护性。"""

        # 测试性能相关
        elif "性能" in prompt_str or "performance" in prompt_lower or "优化" in prompt_str:
            return """性能优化建议：

1. 数据库优化：使用索引、优化查询语句
2. 缓存策略：Redis缓存热点数据
3. 异步处理：使用异步IO提高并发能力
4. 代码优化：避免重复计算，使用高效算法
5. 资源管理：合理设置连接池大小

通过这些优化可以显著提升系统性能。"""

        # 默认响应
        else:
            return f"""感谢您的提问！这是一个模拟的DeepSeek响应。

您的问题是：{prompt_str[:100]}...

基于RAG检索到的文档内容，我为您提供了相关的解答。在实际生产环境中，这个响应将由DeepSeek API根据检索到的上下文实时生成。

[模拟模式 - 第{self._call_count}次调用]"""
    
    def simulate_error(self, error_type: str = "api_error"):
        """
        模拟错误场景，用于测试错误处理
        
        Args:
            error_type: 错误类型
                - "api_error": API调用错误
                - "timeout": 超时错误
                - "rate_limit": 速率限制错误
                - "invalid_key": API密钥无效
        """
        self._call_count += 1
        logger.warning(f"[MOCK] 模拟错误 - 类型: {error_type}")
        
        error_messages = {
            "api_error": "API调用失败，请检查网络连接",
            "timeout": "请求超时，请稍后重试",
            "rate_limit": "API调用速率超限，请降低请求频率",
            "invalid_key": "API密钥无效，请检查配置"
        }
        
        raise Exception(error_messages.get(error_type, "未知错误"))
    
    def reset_call_count(self):
        """重置调用计数"""
        self._call_count = 0
        logger.info("[MOCK] 调用计数已重置")
    
    def get_call_count(self) -> int:
        """获取调用次数"""
        return self._call_count
