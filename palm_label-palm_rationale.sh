RUN_NAME=palm_label-palm_rationale
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR

python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type task_prefix \
    --label_type llm \
    --llm palm \
    --alpha 0.5 \
    --batch_size 16 \
    --grad_steps 4 \
| tee -a $OUTPUT_DIR/train.log