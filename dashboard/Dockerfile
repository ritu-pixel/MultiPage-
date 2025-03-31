FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (adjust if needed)
COPY . .

# Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
