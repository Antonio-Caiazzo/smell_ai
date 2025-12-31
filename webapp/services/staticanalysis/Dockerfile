FROM python:3.11-slim

WORKDIR /app

COPY webapp/services/staticanalysis/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY webapp /app/webapp
# Copy shared components required by Static Analysis
COPY components /app/components
COPY code_extractor /app/code_extractor
COPY detection_rules /app/detection_rules
COPY obj_dictionaries /app/obj_dictionaries

ENV PYTHONPATH=/app

CMD ["uvicorn", "webapp.services.staticanalysis.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
