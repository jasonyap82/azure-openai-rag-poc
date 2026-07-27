#!/usr/bin/env bash
# Provision the minimum Azure footprint for this POC.
# Everything lands in ONE resource group so teardown.sh can delete it all at once.
set -euo pipefail

RG="${RG:-rg-rag-poc}"
LOC="${LOC:-eastus}"
AOAI="${AOAI:-aoai-ragpoc-$RANDOM}"

echo "Resource group : $RG ($LOC)"
echo "Azure OpenAI   : $AOAI"

az group create -n "$RG" -l "$LOC" -o none

az cognitiveservices account create \
  -n "$AOAI" -g "$RG" -l "$LOC" \
  --kind OpenAI --sku S0 \
  --custom-domain "$AOAI" -o none

# Deployment SKUs: GlobalStandard is pay-per-token. Do NOT use ProvisionedManaged --
# it bills reserved capacity hourly and is wildly wrong for a POC.
az cognitiveservices account deployment create \
  -n "$AOAI" -g "$RG" \
#  --deployment-name gpt-4o-mini \
#  --model-name gpt-4o-mini --model-version "2024-07-18" --model-format OpenAI \
  --deployment-name gpt-5-mini \
  --model-name gpt-5-mini --model-version "2025-08-07" --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 10 -o none

az cognitiveservices account deployment create \
  -n "$AOAI" -g "$RG" \
  --deployment-name text-embedding-3-small \
  --model-name text-embedding-3-small --model-version "1" --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 10 -o none

# Grant the signed-in user data-plane access, so the app can use Entra ID, not keys.
USER_OID=$(az ad signed-in-user show --query id -o tsv)
SCOPE=$(az cognitiveservices account show -n "$AOAI" -g "$RG" --query id -o tsv)
az role assignment create \
  --assignee "$USER_OID" \
  --role "Cognitive Services OpenAI User" \
  --scope "$SCOPE" -o none

ENDPOINT=$(az cognitiveservices account show -n "$AOAI" -g "$RG" --query properties.endpoint -o tsv)
echo
echo "Done. Add this to your .env:"
echo "  AZURE_OPENAI_ENDPOINT=$ENDPOINT"
echo
echo "Standing cost of this footprint: \$0 (Azure OpenAI is pay-per-token only)."
echo "Tear down with: RG=$RG ./infra/teardown.sh"
