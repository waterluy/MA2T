#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-28597}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/
# echo $WORK_DIR  ./projects/work_dirs/stage1_track_map/base_track_map/

# Extract --ckpt-folder argument
CKPT_FOLDER=""
for arg in "$@"; do
  if [[ $arg == --ckpt-folder ]]; then
    CKPT_FOLDER=1
  elif [[ $CKPT_FOLDER == 1 ]]; then
    CKPT_FOLDER=$arg
    break
  fi
done

if [ -n "$CKPT_FOLDER" ]; then
  LOG_FILE="${WORK_DIR}${CKPT_FOLDER}/adv_train.$T"
else
  LOG_FILE="${WORK_DIR}logs/adv_train.$T"
fi

if [ ! -d $(dirname "$LOG_FILE") ]; then
    mkdir -p $(dirname "$LOG_FILE")
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/adv_train.py \
    $CFG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee $LOG_FILE