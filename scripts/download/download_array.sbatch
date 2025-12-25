#!/usr/bin/env bash
#SBATCH --job-name=celljepa-dl
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

MANIFEST=${MANIFEST:?Set MANIFEST to a TSV file}
DEST_ROOT=${DEST_ROOT:-data/raw}

line_num=$((SLURM_ARRAY_TASK_ID + 1))
line=$(sed -n "${line_num}p" "$MANIFEST" | sed 's/\r$//')
if [ -z "$line" ]; then
  echo "No line for index ${SLURM_ARRAY_TASK_ID} in $MANIFEST"
  exit 1
fi

url=$(echo "$line" | awk -F'\t' '{print $1}')
relpath=$(echo "$line" | awk -F'\t' '{print $2}')
checksum=$(echo "$line" | awk -F'\t' '{print $3}')

out="${DEST_ROOT}/${relpath}"
mkdir -p "$(dirname "$out")"

echo "Downloading: $url"
echo "To: $out"

CURL_FLAGS=(-L --retry 5 --retry-delay 5 --fail --continue-at -)
if [ "${CURL_INSECURE:-0}" = "1" ]; then
  CURL_FLAGS+=(-k)
fi

curl "${CURL_FLAGS[@]}" -o "$out" "$url"

if [ -n "$checksum" ]; then
  algo=${checksum%%:*}
  hex=${checksum#*:}
  if [ "$algo" = "md5" ]; then
    echo "${hex}  ${out}" | md5sum -c -
  elif [ "$algo" = "sha256" ]; then
    echo "${hex}  ${out}" | sha256sum -c -
  else
    echo "Checksum algorithm not supported in script: $algo"
  fi
fi
