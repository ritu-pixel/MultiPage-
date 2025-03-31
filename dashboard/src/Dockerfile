FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (for dependency caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY dashboard/src/ ./src/

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py"]
