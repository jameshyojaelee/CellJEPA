#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <manifest.tsv> <job_name> [max_parallel]" >&2
  exit 1
fi

MANIFEST=$1
JOB_NAME=$2
MAX_PARALLEL=${3:-20}

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

LINES=$(grep -cve '^\s*$' "$MANIFEST")
if [ "$LINES" -le 0 ]; then
  echo "Manifest has no entries: $MANIFEST" >&2
  exit 1
fi

PARTITION=${PARTITION:-}
if [ -z "$PARTITION" ]; then
  if sinfo -h -o "%P" | tr ' ' '\n' | grep -q "^io\\*\\?$"; then
    PARTITION="io"
  else
    PARTITION="cpu"
  fi
fi

TIME=${TIME:-48:00:00}
DEST_ROOT=${DEST_ROOT:-data/raw}

ARRAY="0-$((LINES-1))%${MAX_PARALLEL}"

echo "Submitting $JOB_NAME: $LINES files, array $ARRAY, partition $PARTITION, time $TIME"

sbatch \
  --partition="$PARTITION" \
  --time="$TIME" \
  --job-name="$JOB_NAME" \
  --array="$ARRAY" \
  --export=MANIFEST="$MANIFEST",DEST_ROOT="$DEST_ROOT",CURL_INSECURE="${CURL_INSECURE:-0}" \
  scripts/download/download_array.sbatch
