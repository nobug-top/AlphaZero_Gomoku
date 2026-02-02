FROM python:3.12-slim

WORKDIR /app

RUN pip install numpy==1.26.4

COPY src/ .

CMD ["python", "http_server.py"]