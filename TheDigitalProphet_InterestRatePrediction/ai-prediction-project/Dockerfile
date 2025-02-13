# Use OpenJDK base image to avoid installation issues with JDK
FROM openjdk:11-jre-slim

# Set the working directory inside the container
WORKDIR /app

# Install curl and other necessary dependencies
RUN apt-get update -y && \
    apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python and necessary packages
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port 5000 to be accessible from outside the container
EXPOSE 5000

# Run the Flask app
CMD ["python3", "app_testing_modified.py"]
