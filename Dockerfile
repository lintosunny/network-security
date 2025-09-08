FROM python:3.10-slim
WORKDIR /app

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

# Install system dependencies and clean up
RUN apt-get update -y \
    && apt-get install -y awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --locked

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]