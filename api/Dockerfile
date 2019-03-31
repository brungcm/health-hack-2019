FROM python:alpine

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV STATUS_FILE /data/status.json
CMD ["python", "/app/main.py"]