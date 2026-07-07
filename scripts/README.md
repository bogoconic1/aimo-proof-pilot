# Operator Mode

Operator mode lets a long-running `train.py` process poll a command repo, run
commands on each node, and upload live output back to GitHub. Use it when the
cluster does not expose SSH directly.

## Start Operators On Nodes

Start one operator process inside each container/node. Prefer launching through
`train.py`, not `train_operator.py`, because `train.py` fetches the newest
`aimo-proof-pilot` code before entering operator mode.

Example:

```bash
GITHUB_TOKEN="$GITHUB_TOKEN" \
HF_TOKEN="$HF_TOKEN" \
python /app/train.py \
  --fetch-update \
  --submissions-repo https://github.com/nguyen599/aimo-proof-pilot.git \
  --submissions-ref main \
  --backend open_instruct_wrapper \
  --model_path /tmp/unused \
  --dataset_path /tmp/unused \
  --output_path /tmp/olmo_train_logs/operator_github_command \
  --logdir /tmp/olmo_train_logs/operator_github_command \
  --operator_mode true \
  --operator_backend github \
  --operator_command_repo nguyen599/command \
  --operator_command_file command.sh \
  --operator_key_file key.txt \
  --operator_github_command_download_mode raw \
  --operator_github_api_refresh_interval_seconds 10 \
  --operator_poll_interval_seconds 2 \
  --operator_live_upload_interval_seconds 30 \
  --operator_github_output_branch_template 'operator-output-node-{node}' \
  --operator_output_upload_queue_path /tmp/olmo_operator_output_upload.lock \
  --operator_output_upload_queue_timeout_seconds 180
```

Node labels come from `GLOBAL_RANK`, `NODE_RANK`, `SLURM_NODEID`, or `RANK`.
For a six-node cluster, clients should usually pass `--nodes 0,1,2,3,4,5`.

Operator logs on each node are written under the `--logdir`, commonly:

```bash
/tmp/olmo_train_logs/operator_github_command/train.log
/tmp/olmo_train_logs/operator_github_command/operator_restarts/
```

## Send Commands

Run commands from the local checkout:

```bash
export GITHUB_TOKEN="$GITHUB_TOKEN"

python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'hostname && nvidia-smi'
```

For larger jobs, write a `.sh` file and send it:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --file aimo-proof-pilot/operator_commands/prime_rl_opd_3node_full_vocab_dpsk_ctx81920_nodes345.sh
```

The client prints a six-character `command_id`. Outputs for that command are
uploaded as:

```text
output_node<N>_<command_id>.txt
```

By default, outputs are written to per-node branches:

```text
operator-output-node-0
operator-output-node-1
...
```

This avoids most Git push conflicts when many nodes upload at the same time.

## Fetch Logs

Fetch all nodes for one command:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  fetch --node all --command-id f77a76 --out-dir /tmp/operator_fetch_f77a76 --no-print
```

Print only useful tail lines:

```bash
for f in /tmp/operator_fetch_f77a76/output_node*; do
  echo "===== $(basename "$f") ====="
  grep -E 'ERROR|Traceback|finished_utc|return_code|vLLM|Prime-RL|ready|timeout|full-vocab|teacher|policy|orchestrator' "$f" | tail -120
done
```

Tail live output:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  tail --node all --command-id f77a76 --interval 30 --delta-first
```

Expect 60-90 seconds of delay in practice because each node uploads through Git.

## Re-running A Command

The operator command ID is the first six hex characters of the SHA-256 hash of
the uploaded command text. If you need to run the same command again
immediately, add a harmless nonce comment:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command $'hostname && nvidia-smi\n# nonce='"$(date +%s%N)"
```

## Restart Operators

Restart starts a fresh `train.py --operator_mode true` process. The old operator
stays alive but becomes passive if its key no longer matches `key.txt`.

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  restart
```

Use this after pushing fixes to `aimo-proof-pilot` when you need all nodes to
fetch the latest operator code. Do not kill remote-shell or endpoint daemons
unless you explicitly intend to disable that separate control path.

## Stop Or Kill Jobs

Cancel active commands but keep operator mode alive:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  cancel
```

Kill one command by command ID:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  kill f77a76 --node all --wait-seconds 90
```

`kill` targets child process groups for that command. It preserves operator
processes, `operator_client.py`, and `train.py --operator_mode true`.

Stop operator mode completely:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  stop
```

Only use `stop` when you intentionally want operators to exit.

## Troubleshooting

Check active operators:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'echo node=${GLOBAL_RANK:-${NODE_RANK:-none}} host=$(hostname); pgrep -af "train.py .*--operator_mode|train_operator.py" || true'
```

Check training or vLLM processes without killing them:

```bash
python aimo-proof-pilot/scripts/operator_client.py \
  --backend github \
  --repo nguyen599/command \
  --nodes 0,1,2,3,4,5 \
  send --command 'echo node=${GLOBAL_RANK:-${NODE_RANK:-none}} host=$(hostname); ps -eo pid,ppid,etime,stat,cmd | grep -E "[p]rime_rl|[v]llm|[t]rain.py|[t]rain_engine"'
```

If output upload is delayed by Git conflicts, wait for the queue to drain before
sending another diagnostic command. Large log files slow down every fetch; clean
old short or oversized output files from the command repo when needed.

