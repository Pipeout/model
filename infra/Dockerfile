FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1

COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY infra/requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "scripts/main.py"]
