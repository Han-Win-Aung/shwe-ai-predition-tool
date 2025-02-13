# Use the official Python image as a base
FROM python:3.11-slim

# Set non-interactive mode to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (needed for H2O) and other required dependencies
RUN apt-get update -y && \
    apt-get install -y openjdk-11-jre curl && \
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
