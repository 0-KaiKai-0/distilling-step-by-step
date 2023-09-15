RUN_NAME=gpt_rationale_clamp0.7_tanh10
DATASET=cqa
GPT=gpt-3.5-turbo
OUTPUT_DIR=outputs/$DATASET/$GPT/$RUN_NAME

mkdir -p $OUTPUT_DIR

python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type gpt_rationale \
    --label_type gt \
    --batch_size 8 \
    --gpt $GPT \
    --gpt_rate 1 \
| tee -a $OUTPUT_DIR/train.log