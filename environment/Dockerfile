FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY frozen-requirements.txt .
RUN pip install --default-timeout=1000 -i https://pypi.tuna.tsinghua.edu.cn/simple -r frozen-requirements.txt

COPY requirements.txt .
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY templates/ ./templates/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
