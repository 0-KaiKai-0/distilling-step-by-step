MODEL=gpt-3.5-turbo
DATASET=cqa
TASK=rationale
KEY=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
OUTPUT_DIR=datasets/$DATASET/$MODEL/$TASK

mkdir -p $OUTPUT_DIR

python -u gpt.py \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--key $KEY \
| tee -a $OUTPUT_DIR/generate.log
