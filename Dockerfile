# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9.19-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED=1

# Create and change to the app directory
WORKDIR /usr/src/app

# Install system dependencies, including libgl1 for OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application dependency manifests to the container image
COPY requirements.txt ./

# Inform about starting dependency installation
RUN echo "Installing Python dependencies..." && \
    pip install --progress-bar=on --verbose -r requirements.txt

# Copy local code to the container image
RUN echo "Copying application files to the container..."
COPY . ./

ENV FLASK_APP=app.py

EXPOSE 8080

# Inform about app startup
# RUN echo "Setting up Gunicorn to run the web service..."

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
