FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (to cache dependencies)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main application files
COPY app.py .
COPY src/dashboard ./src/dashboard

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
