import openai
import os

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
 
openai.api_key = "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"

# system_intel = "You are a writer, you are supposed to rewrite the question. You cannot modify the label, so that it can still lead to the original right answer. No need to output the answer."
system_intel = "You are a writer. You are provided with a question along with the answer choices. Only one of the answers is right. Explain why the question lead to the corresponding answer and give your answer. Your output must end with 'So the answer is'"
prompt = "Write a blog on how to use GPT-4 with python in a jupyter notebook"


def ask_GPT4(system_intel, prompt, n):
    print(prompt)
    new_questions = []
    print("GPT-4:")
    result = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "system", "content": system_intel},
                                            {"role": "user", "content": prompt}],
                                        temperature=0.5,
                                        max_tokens=256,
                                        n=n)
    for idx in range(n):
        question = result['choices'][idx]['message']['content']
        new_questions.append(question)
        print(idx)
        print(question)
    return(new_questions)

dataset_loader = CQADatasetLoader()
datasets = dataset_loader.load_from_json()
train_dataset = datasets['train']
prompt = train_dataset[0]["input"]
sample_cnt = 5

new_questions = ask_GPT4(system_intel, prompt, sample_cnt)