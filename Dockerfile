FROM python:3.13-slim

WORKDIR /api-circuitscan

COPY lib/* lib/
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY models/* models/

RUN pip install --default-timeout=120 -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr

COPY . .

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "main:app", "-k", "sync", "--timeout", "120"]
