# S6 T7：api 镜像（docs/07 §T7）。权重不进镜像（挂载宿主机 models/）。
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "medevirag.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
