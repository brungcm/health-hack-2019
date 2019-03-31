FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential libsm6 libxext6 libfontconfig1 libxrender1

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]