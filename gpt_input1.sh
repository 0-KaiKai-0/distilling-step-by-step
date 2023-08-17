RUN_NAME=gpt_input
DATASET=cqa
GPT=gpt-4
OUTPUT_DIR=outputs/$DATASET/$GPT/$RUN_NAME

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type gpt_input \
    --label_type gt \
    --batch_size 64 \
    --gpt $GPT \
    --gpt_rate 1 \
| tee -a $OUTPUT_DIR/train1.log