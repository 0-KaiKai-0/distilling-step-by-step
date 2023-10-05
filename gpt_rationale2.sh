RUN_NAME=gpt_rationale_clamp0.2-0.8
DATASET=cqa
GPT=gpt-3.5-turbo
OUTPUT_DIR=outputs/$DATASET/$GPT/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp gpt_rationale2.sh $OUTPUT_DIR/run.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type gpt_rationale \
    --label_type gt \
    --batch_size 16 \
    --gpt $GPT \
    --gpt_rate 1 \
    --output_dir $OUTPUT_DIR \
    --sample_loss True \
| tee -a $OUTPUT_DIR/train.log