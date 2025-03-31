FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (to cache dependencies)
COPY requirements.txt .  # Ensure this file exists in the root
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .  # Copies all files (adjust if needed)

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]  # Ensure app.py is in the root or adjust path
