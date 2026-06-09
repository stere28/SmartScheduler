# ── Stage: Builder ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for ortools / some transitive deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage: Runtime ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# curl is used by entrypoint.sh to health-check Ollama and pull models
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Entrypoint script handles Ollama health-check and model pull
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
