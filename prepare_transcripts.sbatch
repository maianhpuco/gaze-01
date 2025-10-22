#!/bin/bash
#SBATCH --job-name=prep_transcripts
#SBATCH --output=logs/prepare_transcripts_%j.out
#SBATCH --error=logs/prepare_transcripts_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_DIR="${SUBMIT_DIR}/logs"
CONFIG_PATH="${SUBMIT_DIR}/config/data_egd-cxr.yaml"

mkdir -p "${LOG_DIR}"

echo "Running prepare_transcripts.py"
echo "  Config path : ${CONFIG_PATH}"

cd "${SUBMIT_DIR}"
python "${SUBMIT_DIR}/prepare_transcripts.py" --config-path "${CONFIG_PATH}"

echo "Transcript preprocessing completed."
