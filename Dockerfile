# Gunakan image Python resmi
FROM python:3.10-slim

# Install git untuk clone repo
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/username/efw-ews.git

# Set working directory ke folder project
WORKDIR /efw-ews

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Perintah default: jalankan aplikasi
CMD ["python", "app/app.py"]
