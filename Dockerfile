FROM python:3.9-slim

WORKDIR /app

# First copy ONLY requirements.txt
COPY requirements.txt .

# Then install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Then copy everything else
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/src/app.py"]
