#!/usr/bin/env bash
set -euo pipefail

# Submit the exact operator payload that succeeded as command_id=08deb7.
python scripts/operator_client.py send \
  --file operator_commands/phase1_32b_tp8_pp2_seq92160_pdbs6_ga6.txt

# Optional: list/fetch logs for the successful run after submission.
# Replace 08deb7 with the new command_id printed by the send command above.
python scripts/operator_client.py list --command-id 08deb7
python scripts/operator_client.py fetch \
  --node all \
  --command-id 08deb7 \
  --out-dir /tmp/operator_08deb7 \
  --no-print
