# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Rust and other build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create run.py if it doesn't exist
RUN if [ ! -f run.py ]; then echo 'import uvicorn\n\nif __name__ == "__main__":\n    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)' > run.py; fi

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py"] 