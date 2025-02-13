# Use the official Python image as a base
FROM python:3.11

# Set non-interactive mode to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install curl, gnupg2, and dependencies for OpenJDK 11
RUN apt-get update -y && \
    apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    openjdk-11-jre && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY TheDigitalProphet_InterestRatePrediction/ai-prediction-project/requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY TheDigitalProphet_InterestRatePrediction/ai-prediction-project /app

# Expose the port that your app will run on (adjust if needed)
EXPOSE 5000

# Set the command to run your app (replace app_testing_modified.py with your main Python file)
CMD ["python", "app_testing_modified.py"]
