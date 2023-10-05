RUN_NAME=palm_label-palm_rationale
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp palm_label-palm_rationale.sh $OUTPUT_DIR/run.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type task_prefix \
    --label_type llm \
    --llm palm \
    --alpha 0.5 \
    --batch_size 16 \
    --output_dir $OUTPUT_DIR \
| tee -a $OUTPUT_DIR/train.log