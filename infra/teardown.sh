#!/usr/bin/env bash
# Delete everything. Run this the moment you're finished.
set -euo pipefail
RG="${RG:-rg-rag-poc}"

echo "This will delete resource group '$RG' and everything in it."
read -rp "Type the resource group name to confirm: " confirm
[[ "$confirm" == "$RG" ]] || { echo "Aborted."; exit 1; }

az group delete -n "$RG" --yes --no-wait
echo "Deletion started (running in background). Verify with: az group list -o table"
