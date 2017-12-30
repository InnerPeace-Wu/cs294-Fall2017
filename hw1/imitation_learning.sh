#!/usr/bin/env bash

set -x
set -e

envname=$1

LOG="output/bc_log.txt.`date +%Y-%m-%d_%H-%M-%S`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python imitation_learning.py \
    $1 \
    --batch_size 32 \
    --epoch 100
