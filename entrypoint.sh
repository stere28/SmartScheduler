#!/bin/sh
# ──────────────────────────────────────────────────────────────────────────────
# SmartScheduler – Container Entrypoint
#
# 1. Wait for the Ollama service to become available (health-check loop).
# 2. Pull the configured model if not already downloaded.
# 3. Execute the SmartScheduler pipeline.
# ──────────────────────────────────────────────────────────────────────────────

set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MODEL="${LLM_MODEL:-llama3.2}"
MAX_WAIT=120   # seconds
INTERVAL=3

echo "═══════════════════════════════════════════════════"
echo "  SmartScheduler – Startup"
echo "  Ollama URL : ${OLLAMA_URL}"
echo "  LLM Model  : ${MODEL}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Wait for Ollama ───────────────────────────────────────────────────
echo ""
echo "[1/3] Waiting for Ollama to be ready…"
elapsed=0
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "  ✗ Ollama did not start within ${MAX_WAIT}s. Aborting."
        exit 1
    fi
    echo "  … Ollama not yet ready (${elapsed}s elapsed). Retrying in ${INTERVAL}s…"
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done
echo "  ✓ Ollama is ready."

# ── Step 2: Pull model if needed ──────────────────────────────────────────────
echo ""
echo "[2/3] Checking model '${MODEL}'…"

# Ollama API returns "name":"llama3.2:latest" – strip the tag for matching
MODEL_BASE=$(echo "${MODEL}" | cut -d: -f1)
if curl -sf "${OLLAMA_URL}/api/tags" | grep -q "\"${MODEL_BASE}:"; then
    echo "  ✓ Model '${MODEL}' is already available."
else
    echo "  ↓ Pulling model '${MODEL}' (this may take several minutes)…"
    curl -sf "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${MODEL}\"}" \
        --no-buffer | while IFS= read -r line; do
            echo "    ${line}"
        done
    echo "  ✓ Model '${MODEL}' pulled successfully."
fi

# ── Step 3: Run the pipeline ──────────────────────────────────────────────────
echo ""
echo "[3/3] Launching SmartScheduler pipeline…"
echo ""

exec python main.py "$@"
