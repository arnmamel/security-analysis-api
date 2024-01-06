# Use an official Python runtime as a parent image
FROM python:3.10.13

# Install Git
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /usr/src/sec-ai

# Copiem l'entorn virtual, aplicació i arxius necessaris
COPY tfgrcaai /usr/src/sec-ai/tfgrcaai
COPY app /usr/src/sec-ai/app
COPY logs /usr/src/sec-ai/logs
COPY datasets /usr/src/sec-ai/datasets
COPY tmp /usr/src/sec-ai/tmp
COPY dep /usr/src/sec-ai/dep
COPY requirements.txt /usr/src/sec-ai/

# Install dependencies from the first repository
RUN /bin/bash -c "source /usr/src/sec-ai/tfgrcaai/bin/activate && pip install --no-cache-dir -r /usr/src/sec-ai/requirements.txt"

# Expose the necessary ports
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
