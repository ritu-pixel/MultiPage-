FROM python:3.9-slim

WORKDIR /app

# Install system libs for matplotlib/evidently
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the dashboard folder (ignores other files)
COPY dashboard ./dashboard

# Set Python path for imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
