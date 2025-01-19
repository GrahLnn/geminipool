# 选择一个轻量级基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件并安装依赖
COPY pyproject.toml .
RUN pip install uv && uv pip install -e .

# 再将剩余代码复制进去
COPY . .

# 容器启动时执行的命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
