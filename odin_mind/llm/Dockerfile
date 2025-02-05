FROM ubuntu:20.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_dev.txt .
RUN pip3 install --no-cache-dir -r requirements_dev.txt

# Set the working directory
WORKDIR /app

# Copy the code
COPY . /app

# Mount the models directory from the host system
VOLUME ["/app/llm/models"]

# Expose the necessary port (adjust if needed)
EXPOSE 5000

# Command to run the application (adjust according to your LLM service)
CMD ["python3", "llm/scripts/llm_startup.py"]
