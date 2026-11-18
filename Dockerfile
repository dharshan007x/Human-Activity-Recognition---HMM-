FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and resources
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

