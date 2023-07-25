RUN_NAME=standard_distill
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR

python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type standard \
    --label_type llm \
    --llm palm \
    --batch_size 64 \
| tee -a $OUTPUT_DIR/train.log