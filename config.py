"""
配置文件 - 支持测试环境和生产环境切换
"""

import os
from enum import Enum

class Environment(Enum):
    """环境枚举"""
    PRODUCTION = "production"
    TEST = "test"
    MOCK = "mock"

class Config:
    """配置类"""
    
    # 环境设置 - 通过环境变量控制，默认为MOCK
    ENV = Environment(os.getenv("APP_ENV", "mock").lower())
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_TEMPERATURE = 0.1
    
    # 数据路径
    DATA_PATH = "./data"
    MODEL_PATH = "./model_cache/BAAI/bge-small-zh-v1___5"
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def is_production(cls):
        """判断是否为生产环境"""
        return cls.ENV == Environment.PRODUCTION
    
    @classmethod
    def is_test(cls):
        """判断是否为测试环境"""
        return cls.ENV == Environment.TEST
    
    @classmethod
    def is_mock(cls):
        """判断是否为模拟环境"""
        return cls.ENV == Environment.MOCK
    
    @classmethod
    def get_env_name(cls):
        """获取环境名称"""
        return cls.ENV.value.upper()
