FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py drift.sql label.sql maint_pipeline.json ./
ENTRYPOINT ["python", "main.py"]