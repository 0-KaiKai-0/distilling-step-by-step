RUN_NAME=gpt_rationale_alpha0.7_lr5e-5
DATASET=cqa
GPT=gpt-3.5-turbo
OUTPUT_DIR=outputs/$DATASET/$GPT/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp gpt_rationale1.sh $OUTPUT_DIR/run.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run.py \
    --from_pretrained google/t5-v1_1-base \
    --dataset $DATASET \
    --model_type gpt_rationale \
    --label_type gt \
    --batch_size 16 \
    --gpt $GPT \
    --gpt_rate 1 \
    --alpha 0.7 \
    --lr 5e-5 \
    --output_dir $OUTPUT_DIR \
| tee -a $OUTPUT_DIR/train.log