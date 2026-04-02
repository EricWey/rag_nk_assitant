#!/bin/bash

# Streamlit Web 应用启动脚本

echo "正在启动 RAG 专家助手 Web 应用..."
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements_web.txt

# 启动应用
echo "启动 Streamlit 应用..."
echo ""
echo "访问地址: http://localhost:8501"
echo ""

streamlit run web_app.py
