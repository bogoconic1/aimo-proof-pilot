set -euo pipefail

LOGS=/tmp/imochallenge/logs
RUN=/tmp/imochallenge/run
PYTHON_SITE=/tmp/imochallenge/python_site
mkdir -p "$LOGS" "$RUN" "$PYTHON_SITE"

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
HOST="$(hostname 2>/dev/null || echo unknown-host)"
PY="$(command -v python3 || command -v python || echo /usr/bin/python3)"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export CLIENT_ID="${CLIENT_ID:-node${NODE_LABEL}-${HOST}}"

echo "remote-shell daemon hotfix node=${NODE_LABEL} host=${HOST}"
echo "python=${PY}"
echo "python_site=${PYTHON_SITE}"
echo "hf_token_present=$([ -n "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ] && echo yes || echo no)"

if ! "$PY" - <<'PY'
import gradio_client
print("gradio_client import OK", getattr(gradio_client, "__version__", "unknown"))
PY
then
    echo "installing gradio_client into user base"
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m ensurepip --user || true
    fi
    "$PY" -m pip install --target "$PYTHON_SITE" --upgrade --break-system-packages --no-cache-dir "gradio_client>=1.3"
fi

"$PY" - <<'PY'
import gradio_client
print("gradio_client import OK", getattr(gradio_client, "__version__", "unknown"))
PY

if pgrep -af "python.* /app/remote-shell/daemon/client.py" | grep -v "operator_client" | grep -v "grep" >/tmp/remote-shell-client-existing.txt; then
    echo "remote-shell daemon already running:"
    cat /tmp/remote-shell-client-existing.txt
    exit 0
fi

cat > "$RUN/start-relay-daemon-fixed.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
PYTHON_SITE=/tmp/imochallenge/python_site
PY="${PY:-$(command -v python3 || command -v python || echo /usr/bin/python3)}"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
HOST="$(hostname 2>/dev/null || echo unknown-host)"
export CLIENT_ID="${CLIENT_ID:-node${NODE_LABEL}-${HOST}}"
exec "$PY" /app/remote-shell/daemon/client.py
EOF
chmod +x "$RUN/start-relay-daemon-fixed.sh"

echo "starting remote-shell daemon without touching existing entrypoint/operator processes"
setsid nohup "$RUN/start-relay-daemon-fixed.sh" >> "$LOGS/relay-daemon-fix.log" 2>&1 < /dev/null &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$RUN/relay-daemon-fix.pid"
sleep 3

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "remote-shell daemon started pid=${DAEMON_PID}"
else
    echo "remote-shell daemon exited during startup; tail follows"
    tail -80 "$LOGS/relay-daemon-fix.log" || true
    exit 1
fi

echo "process check:"
pgrep -af "python.* /app/remote-shell/daemon/client.py" || true
echo "log tail:"
tail -60 "$LOGS/relay-daemon-fix.log" || true
