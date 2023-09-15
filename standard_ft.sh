RUN_NAME=standard_ft
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=2,3 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type standard \
    --label_type gt \
    --batch_size 32 \
| tee -a $OUTPUT_DIR/train.log