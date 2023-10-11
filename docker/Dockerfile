# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7-slim

RUN apt-get update && apt-get install -y libxrender1 libxext6

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install 

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /spectral_network_generator
