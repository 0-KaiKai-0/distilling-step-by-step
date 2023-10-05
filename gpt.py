import openai
import os
import json
import argparse
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from data_utils import CQADatasetLoader


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--key', type=str)
args = parser.parse_args()

openai.api_key = args.key

if args.task == 'question':
    system_intel = "You are a writer, you are supposed to rewrite only the question. Make sure it can still lead to the original right answer. Output only the generated question and do not output the answer."
elif args.task == 'rationale':
    system_intel = "You are a writer. You are provided with a question along with the answer choices. Only one of the answers is right. Explain why the question lead to the corresponding answer and give your answer. Your output must end with 'So the answer is' and be no more than 90 tokens."
# system_intel = "You are a writer. You are provided with a question along with its label. Explain why the question lead to the corresponding label"

sample_cnt = 3

@retry(wait=wait_random_exponential(min=1, max=10))
def get_questions(system_intel, prompt, n):
    # import pdb
    # pdb.set_trace()
    questions = []
    result = openai.ChatCompletion.create(model=args.model,
                                        messages=[{"role": "system", "content": system_intel},
                                            {"role": "user", "content": prompt}],
                                        temperature=1,
                                        max_tokens=1024,
                                        n=n)
    for idx in range(n):
        question = result['choices'][idx]['message']['content']
        if '\nAnswer Choices:' not in question:
            question = question + '\nAnswer Choices:' + prompt.split('\nAnswer Choices:')[1]
        questions.append(question)
    
    new_texts.append(questions)
    if q % 50 == 0:
        print(prompt)
        print(args.model)
        for question in questions:
            print(question)


@retry(wait=wait_random_exponential(min=1, max=5))
def get_rationales(system_intel, prompt, n):
    rationales = []
    
    while len(rationales) < n:
        result = openai.ChatCompletion.create(model=args.model,
                                            messages=[{"role": "system", "content": system_intel},
                                                {"role": "user", "content": prompt}],
                                            temperature=1,
                                            max_tokens=96,
                                            n=n)
        for idx in range(n):
            rationale = result['choices'][idx]['message']['content']
            if 'So the answer is' in rationale:
                rationales.append(rationale)
        # import pdb
        # pdb.set_trace()

    rationales = rationales[:n]
        

    new_texts.append(rationales)
    if q % 50 == 0:
        print(prompt)
        print(args.model)
        for rationale in rationales:
            print(rationale)

# with open("/root/workspace/distilling-step-by-step/datasets/cqa/gpt4/questions/train_question_0.json") as f:
#     output = json.load

dataset_loader = CQADatasetLoader()
datasets = dataset_loader.load_from_json()
splits = ['train', 'test']
for split in splits:
    train_dataset = datasets[split]
    output_dir = "datasets/" +  args.dataset + "/" + args.model + "/" + args.task
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if split == 'test':
        part_cnt = 2
    else:
        part_cnt = 10
    part_num = int(len(train_dataset) / part_cnt)
    for p in range(part_cnt):
        start = p * part_num
        end = start + part_num
        if p == part_cnt - 1:
            end = len(train_dataset)
        new_texts = []
        for q in range(start, end):
            t0 = time.time()
            prompt = train_dataset[q]["input"]
            if args.task == 'question':
                get_questions(system_intel, prompt, sample_cnt)
            elif args.task == 'rationale':
                get_rationales(system_intel, prompt, sample_cnt)
            
            t1 = time.time()
            print(f"{args.task} {q} generated in {(t1-t0):.2f} seconds")
        with open(f"{output_dir}/{split}_{args.task}_{p}.json", 'w', encoding='utf-8') as file_obj:
            json.dump(new_texts, file_obj, ensure_ascii=False)