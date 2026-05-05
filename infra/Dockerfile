FROM python:3.10-slim

WORKDIR /training-app

COPY scripts/ scripts/
COPY configs/ configs/
COPY requirements/ requirements/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /training-app/requirements/requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1

EXPOSE 8081

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8081"]
