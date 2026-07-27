#!/usr/bin/env bash
# OPTIONAL: add Azure AI Search to demo hybrid retrieval + semantic reranking.
#
# COST WARNING: unlike Azure OpenAI, a search service bills by the HOUR whether or not
# you query it. Basic is roughly $0.10/hour (~$74/month). Provision it, capture your
# eval comparison, and run teardown.sh the same day.
#
#   SKU=free  -> $0, but NO semantic ranker (hybrid/BM25 still work)
#   SKU=basic -> ~$0.10/hour, semantic ranker available (billed separately)
set -euo pipefail

RG="${RG:-rg-rag-poc}"
LOC="${LOC:-eastus}"
SKU="${SKU:-free}"
SEARCH="${SEARCH:-srch-ragpoc-$RANDOM}"

if [[ "$SKU" != "free" ]]; then
  read -rp "SKU=$SKU bills hourly until deleted. Continue? [y/N] " ok
  [[ "$ok" == "y" ]] || exit 1
fi

az search service create -n "$SEARCH" -g "$RG" -l "$LOC" --sku "$SKU" -o none

USER_OID=$(az ad signed-in-user show --query id -o tsv)
SCOPE=$(az search service show -n "$SEARCH" -g "$RG" --query id -o tsv)
for ROLE in "Search Service Contributor" "Search Index Data Contributor"; do
  az role assignment create --assignee "$USER_OID" --role "$ROLE" --scope "$SCOPE" -o none
done

echo
echo "Add to .env:"
echo "  RETRIEVER=azure_search"
echo "  AZURE_SEARCH_ENDPOINT=https://$SEARCH.search.windows.net"
echo
echo "Then: python -m src.ingest && python -m eval.evaluate"
echo "DELETE IT WHEN DONE: RG=$RG ./infra/teardown.sh"
