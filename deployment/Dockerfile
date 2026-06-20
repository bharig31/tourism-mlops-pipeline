# Dockerfile for Wellness Tourism Package Prediction Deployment
# This Dockerfile sets up a containerized environment for the Streamlit application

# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir xgboost==2.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY utils.py .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
