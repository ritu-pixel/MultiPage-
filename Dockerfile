FROM python:3.9-slim

WORKDIR /app

# Install system libs first
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
