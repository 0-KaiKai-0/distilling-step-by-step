RUN_NAME=gt_label-palm_rationale
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp gt_label-palm_rationale.sh $OUTPUT_DIR/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type task_prefix \
    --label_type gt \
    --llm palm \
    --alpha 0.5 \
    --batch_size 32 \
    --output_dir $OUTPUT_DIR \
| tee -a $OUTPUT_DIR/train.log