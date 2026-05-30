FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "src.infrastructure.adapters.in_.main:app", "--host", "0.0.0.0", "--port", "8000"]
