FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY artifacts/ ./artifacts/

EXPOSE 8501

ENV PORT=8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "2", "--timeout", "120", "app:app"]
