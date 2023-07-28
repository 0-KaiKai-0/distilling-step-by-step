RUN_NAME=gpt_input
DATASET=cqa
GPT=gpt-3.5-turbo
OUTPUT_DIR=outputs/$DATASET/$GPT/$RUN_NAME

mkdir -p $OUTPUT_DIR

python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type gpt_input \
    --label_type gt \
    --batch_size 16 \
    --grad_steps 4 \
    --gpt $GPT \
    --gpt_rate 4 \
| tee -a $OUTPUT_DIR/train.log