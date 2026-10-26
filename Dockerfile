# The Multimodal RAG Engine - Week 9 project
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mmrag/ ./mmrag/
COPY data/ ./data/
COPY config/ ./config/
COPY scripts/ ./scripts/

EXPOSE 8000
# DEMO_MODE=1 by default so the container runs with no keys; override at `docker run -e`.
ENV DEMO_MODE=1
CMD ["uvicorn", "--factory", "mmrag.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
