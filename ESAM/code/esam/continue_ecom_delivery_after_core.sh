#!/usr/bin/env bash
set -eo pipefail
cd /mnt/ssd/users/prithvi/deepLearning/TabM/paper

while pgrep -af "run_experiment_esam.py --dataset (homesite-insurance|cooking-time|sberbank-housing)" >/dev/null 2>&1; do
  sleep 60
done

nohup env DATASETS_GPU0=NONE DATASETS_GPU1="ecom-offers delivery-eta" COMPACT=0 \
  bash esam/run_safe_esam_seed42_screen_5var_parallel.sh \
  > /mnt/ssd/users/prithvi/deepLearning/logs/esam/safe_esam_launch_ecom_delivery.log 2>&1 &

echo "started_continuation_pid=$!"
