FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project structure
COPY . .

# Set Python path to include dashboard/src
ENV PYTHONPATH="${PYTHONPATH}:/app/dashboard/src"

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
