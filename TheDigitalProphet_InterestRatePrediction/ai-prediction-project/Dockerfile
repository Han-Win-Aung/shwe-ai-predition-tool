# Use the official Python image as a base
FROM python:3.11-slim

# Install Java (needed for H2O)
RUN apt-get update && apt-get install -y openjdk-11-jre

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY TheDigitalProphet_InterestRatePrediction/ai-prediction-project/requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the entire application code into the container
COPY TheDigitalProphet_InterestRatePrediction/ai-prediction-project /app

# Expose the port that your app will run on (adjust if needed)
EXPOSE 5000

# Set the command to run your app (replace app.py with your main Python file)
CMD ["python", "app_testing_modified.py"]  # Replace app.py with your main entry file
