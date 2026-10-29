# Use a lightweight official Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for compiling packages like XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (this leverages Docker's layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (app.py, consumer.py, models/, etc.) into the container
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Default command to start the FastAPI inference server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
