FROM python:3.11-slim

WORKDIR /app

COPY webapp/services/report/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp /app/webapp
ENV PYTHONPATH=/app

CMD ["uvicorn", "webapp.services.report.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
