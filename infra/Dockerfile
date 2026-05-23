FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1

COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY infra/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8081

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8081"]
