RUN_NAME=standard_distill
DATASET=cqa
OUTPUT_DIR=outputs/$DATASET/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp standard_distill.sh $OUTPUT_DIR/run.sh

python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type standard \
    --label_type llm \
    --llm palm \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR \
| tee -a $OUTPUT_DIR/train.log