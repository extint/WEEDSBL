#!/bin/bash

# # Step 1: Train Teacher (RGB+NIR)
# echo "=========================================="
# echo "Training Teacher Model (RGB+NIR)"
# echo "=========================================="

# python3 scripts/train_weedsgalore.py \
#     --model lightmanet \
#     --use_rgbnir \
#     --augment \

# # Check teacher training success
# if [ $? -ne 0 ]; then
#     echo "ERROR: Teacher training failed!"
#     exit 1
# fi

# # Find the best teacher checkpoint (most recent experiment)
TEACHER_CHECKPOINT=$(find ./experiments_weedsgalore -name "best_model.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "ERROR: Could not find teacher checkpoint!"
    exit 1
fi

echo "=========================================="
echo "Teacher Complete! Checkpoint:"
echo "$TEACHER_CHECKPOINT"
echo "=========================================="

# Step 2: Train Student with Distillation (RGB only)
echo "=========================================="
echo "Training Student with Distillation (RGB)"
echo "=========================================="

# python3 scripts/distill_train_weedsgalore.py \
#     --teacher_checkpoint "$TEACHER_CHECKPOINT" \
#     --student_model unet \
#     --augment

# Find the latest student checkpoint
LATEST_STUDENT=$(find ./distill_experiments_weedsgalore -name "latest_student.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

if [ -n "$LATEST_STUDENT" ]; then
    echo "Resuming from: $LATEST_STUDENT"
    python3 scripts/distill_train_weedsgalore.py \
        --teacher_checkpoint "$TEACHER_CHECKPOINT" \
        --student_model unet \
        --data_root /home/vedantmehra/Downloads/weedsgalore-dataset \
        --augment \
        --num_workers 0 \
        --resume "$LATEST_STUDENT"
else
    echo "No checkpoint found, starting from scratch"
    python3 scripts/distill_train_weedsgalore.py \
        --teacher_checkpoint "$TEACHER_CHECKPOINT" \
        --student_model lightmanet \
        --data_root /home/vedantmehra/Downloads/weedsgalore-dataset \
        --augment \
        --num_workers 0
fi


# Check distillation success
if [ $? -ne 0 ]; then
    echo "ERROR: Distillation failed!"
    exit 1
fi

echo "=========================================="
echo "Pipeline Complete!"
echo "Teacher: $TEACHER_CHECKPOINT"
echo "=========================================="

