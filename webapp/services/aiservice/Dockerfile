FROM python:3.11-slim

WORKDIR /app

COPY webapp/services/aiservice/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp /app/webapp
ENV PYTHONPATH=/app

CMD ["uvicorn", "webapp.services.aiservice.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
