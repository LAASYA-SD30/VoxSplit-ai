FROM python:3.11-slim

# Install ffmpeg (your app needs it)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch (much smaller, fits free tier)
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir flask flask-cors gunicorn soundfile

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
