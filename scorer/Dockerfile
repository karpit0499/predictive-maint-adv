FROM python:3.12-slim
WORKDIR /app
# Python block-buffers stdout when it isn't attached to a TTY (always the case in a
# container). Without this, short log lines sit in the buffer and never reach Cloud
# Logging — the Phase 9 HIGH_RISK alert silently never fires.
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app