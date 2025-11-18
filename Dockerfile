# Use an official Python 3.11 image (wheels available)
FROM python:3.11-slim

# metadata
LABEL maintainer="Anshu"

# set working directory
WORKDIR /app

# system deps required for some wheels / optional builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# copy only requirements first for caching
COPY requirements.txt .

# upgrade packaging tools and install requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . .

# Ensure model files are readable (if included in repo)
RUN chmod -R a+r /app

# expose the port that Render / Docker will use
ENV PORT=5000
EXPOSE 5000

# use a lightweight production server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "3", "--threads", "3", "--timeout", "120"]
