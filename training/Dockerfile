FROM python:3.11-slim

WORKDIR /train

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# Copy training code
COPY . /train

# Create checkpoint directory
RUN mkdir -p /train/checkpoints

ENV PYTHONUNBUFFERED=1

# Default command runs training script
CMD ["python", "urban_world_model.py"]

