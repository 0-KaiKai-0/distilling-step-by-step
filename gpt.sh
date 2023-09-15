MODEL=gpt-3.5-turbo
DATASET=cqa
TASK=rationale
KEY=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
OUTPUT_DIR=datasets/$DATASET/$MODEL/$TASK-pdb

mkdir -p $OUTPUT_DIR

python -u gpt.py \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--key $KEY \
| tee -a $OUTPUT_DIR/pdb.log
