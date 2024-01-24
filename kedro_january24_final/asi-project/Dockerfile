# Use a base image with Python and Kedro dependencies
FROM python:3.8-slim

# Set the working directory
WORKDIR /home/kali/Desktop/kedro/asi-project

# Install build tools (gcc) and other dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file initially
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the Kedro project into the container
COPY . /home/kali/Desktop/kedro/asi-project

# Expose the FastAPI port
EXPOSE 8000

# Set environment variables
ENV KEDRO_HOME=/home/kali/Desktop/kedro/asi-project

# Run Kedro and FastAPI when the container starts
CMD ["bash", "run_kedro.sh"]
