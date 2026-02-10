# Multi-stage build
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir --prefer-binary \
    numpy scipy matplotlib networkx flask gunicorn gevent \
    pytesseract opencv-python-headless ultralytics

# Runtime stage
FROM python:3.13-slim

WORKDIR /api-circuitscan

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY lib lib/
COPY main.py .
COPY models models/

# Set PATH and Python optimization flags
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "main:app", "-k", "sync", "--timeout", "120"]