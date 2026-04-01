"""
测试模拟层的测试代码
验证MockDeepSeek的各种输出情况
"""

import logging
from mock_llm import MockDeepSeek

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mock_llm():
    """测试模拟LLM的各种功能"""
    
    print("=" * 70)
    print("开始测试MockDeepSeek模拟层")
    print("=" * 70)
    
    # 1. 测试初始化
    print("\n[测试1] 初始化MockDeepSeek")
    mock_llm = MockDeepSeek(
        model="deepseek-chat",
        api_key="test_key",
        temperature=0.1
    )
    print(f"✓ 模拟LLM初始化成功")
    print(f"  API密钥: {mock_llm.api_key if mock_llm.api_key else 'N/A'}")
    print(f"  温度: {mock_llm.temperature}")
    
    # 2. 测试不同类型的查询
    print("\n[测试2] 测试不同类型的查询")
    
    test_questions = [
        "这个文档的主要内容是什么？",
        "请给我展示一些代码示例",
        "如何处理代码中的错误？",
        "如何优化系统性能？",
        "一个通用的技术问题"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n  子测试 {i}: {question}")
        try:
            response = mock_llm.predict(question)
            print(f"  ✓ 响应长度: {len(response)} 字符")
            print(f"  响应预览: {response[:50]}...")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    # 3. 测试错误处理
    print("\n[测试3] 测试错误处理")
    
    error_types = ["api_error", "timeout", "rate_limit", "invalid_key"]
    for error_type in error_types:
        try:
            mock_llm.simulate_error(error_type)
            print(f"  ✗ {error_type}: 未能触发预期的错误")
        except Exception as e:
            print(f"  ✓ {error_type}: {str(e)[:40]}...")
    
    # 4. 测试调用统计
    print("\n[测试4] 测试调用统计")
    print(f"  总调用次数: {mock_llm.get_call_count()}")
    
    # 5. 测试重置
    mock_llm.reset_call_count()
    print(f"  重置后调用次数: {mock_llm.get_call_count()}")
    print("  ✓ 调用计数重置成功")
    
    # 6. 测试元数据
    print("\n[测试5] 测试元数据")
    metadata = mock_llm.metadata
    print(f"  模型名称: {metadata['model_name']}")
    print(f"  上下文窗口: {metadata['context_window']}")
    print(f"  ✓ 元数据获取成功")
    
    # 7. 测试批量查询
    print("\n[测试6] 测试批量查询")
    batch_questions = [
        "什么是向量数据库？",
        "如何使用LlamaIndex？",
        "DeepSeek模型的特点是什么？"
    ]
    
    for i, q in enumerate(batch_questions, 1):
        response = mock_llm.predict(q)
        print(f"  查询 {i}: {len(response)} 字符响应")
    
    print(f"\n  批量查询完成，总调用次数: {mock_llm.get_call_count()}")
    
    print("\n" + "=" * 70)
    print("MockDeepSeek测试完成 ✓")
    print("=" * 70)

def test_interface_compatibility():
    """测试接口兼容性"""
    print("\n[兼容性测试] 验证接口与真实API一致")
    
    mock_llm = MockDeepSeek()
    
    # 检查必要的方法是否存在
    required_methods = ['chat', 'predict', 'complete', 'metadata']
    for method in required_methods:
        if hasattr(mock_llm, method):
            print(f"  ✓ 方法 '{method}' 存在")
        else:
            print(f"  ✗ 方法 '{method}' 缺失")
    
    # 检查属性
    required_attrs = ['model', 'temperature', 'api_key']
    for attr in required_attrs:
        if hasattr(mock_llm, attr):
            print(f"  ✓ 属性 '{attr}' 存在")
        else:
            print(f"  ✗ 属性 '{attr}' 缺失")

if __name__ == "__main__":
    test_mock_llm()
    test_interface_compatibility()
