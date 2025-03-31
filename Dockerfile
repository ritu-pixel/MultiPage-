FROM python:3.9-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy ONLY requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest (excluding unnecessary files)
COPY dashboard ./dashboard

# 4. Set Python path (critical for imports)
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
