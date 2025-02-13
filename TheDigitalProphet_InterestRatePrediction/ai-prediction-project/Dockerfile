# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Add Java repository and install dependencies
RUN apt-get update -y && \
    apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates && \
    curl -sL https://packages.openjdk.java.net/openjdk11/ubuntu/ | tee /etc/apt/sources.list.d/openjdk.list && \
    apt-get update -y && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 to be accessible from outside the container
EXPOSE 5000

# Run the Flask app
CMD ["python", "app_testing_modified.py"]
