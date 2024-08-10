FROM python:3

WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/
# Run Flask app
CMD ["python", "app1.py"]
