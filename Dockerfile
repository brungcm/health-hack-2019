FROM tensorflow/tensorflow:latest-gpu

# General dependencies
RUN apt-get update && apt-get install -y \	
	libsm6 \
    libfontconfig1 \
    libxrender1 \
    libxext6

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt