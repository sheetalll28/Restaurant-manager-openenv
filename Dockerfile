# Dockerfile for Hugging Face Spaces deployment
# Build:  docker build -t restaurant-manager .
# Run:    docker run -p 7860:7860 -e HF_TOKEN=your_token restaurant-manager

FROM python:3.11-slim

# Metadata
LABEL maintainer="kajaljotwani"
LABEL description="Restaurant Manager OpenEnv — AI restaurant shift management"
LABEL version="1.1.0"

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching — requirements change less than code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

# Health check — ensures HF marks space as Running
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/').read()" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]