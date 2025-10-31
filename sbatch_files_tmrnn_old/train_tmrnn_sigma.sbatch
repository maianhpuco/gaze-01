#!/bin/bash
#SBATCH --job-name=tmrnn_sigma
#SBATCH --output=logs_sigma/tmrnn_sigma_%j.out
#SBATCH --error=logs_sigma/tmrnn_sigma_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG="configs/egd_cxr_sigma.yaml"
LOG_DIR="logs_sigma"
RUN_ROOT="runs/checkpoints/tmrnn_sigma_${SLURM_JOB_ID}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

TMP_CFG="$(mktemp /tmp/tmrnn_cfg_XXXX.yaml)"
trap 'rm -f "${TMP_CFG}"' EXIT

python - <<'PY' "${CONFIG}" "${RUN_ROOT}" "${TMP_CFG}"
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
tmp_path = Path(sys.argv[3])

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg.setdefault("output_path", {})
cfg["output_path"]["checkpoint_dir"] = str(run_root)

tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "Starting TMRNN training at: $(date)"
echo "Config: ${CONFIG}"
echo "Checkpoint dir: ${RUN_ROOT}"

python main_train.py --config "${TMP_CFG}"

echo "Finished at: $(date)"
