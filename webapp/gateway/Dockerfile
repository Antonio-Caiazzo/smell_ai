FROM python:3.11-slim

WORKDIR /app

COPY webapp/gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything properly
COPY webapp /app/webapp

# Set PYTHONPATH to /app so 'from webapp...' works
ENV PYTHONPATH=/app

CMD ["uvicorn", "webapp.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
