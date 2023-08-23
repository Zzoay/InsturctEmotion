
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse, logging
import pickle

import torch
import torch.nn as nn
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset, Dataset

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import (
    LoraConfig,
    PeftModel,
)

from prompt_wrap import wrap_esc


use_lora = True
load_in_8bit = True
lora_hyperparams_file = 'config/lora.json'
model_name_or_path = '/data/jianggongyao/Llama-2-7b-chat-hf/'
peft_path = '/data/jianggongyao/checkpoints/checkpoint-1350'

data = wrap_esc(split='test')

config = LoraConfig.from_pretrained(peft_path)

tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit = load_in_8bit if use_lora else False,
    device_map='auto',
)

if use_lora:
    model = PeftModel.from_pretrained(model, peft_path, config=config)

model.eval()

task = 'dialogue'
case = False

# print(model)

if task == 'dialogue':
    system_prompt = """You are a digital companion with the ability to understand and empathize with a variety of human emotions.
    Provide emotional comfort and companionship to users through meaningful dialogue.
    Generate responses that affirm and validate the user's feelings, based on the context."""
    user_inputs = [
    "Who are you?",
    "I am so sad.",
    "I'm obsessed with work.",
    "I don't know what shoud I do next",
    "Thanks."
    ]
    user_input = user_inputs[0]
    prompt = system_prompt + f"\nUser: {user_input}\nBot: "
elif task == 'er':
    system_prompt = """You are a compassionate companion bot, understanding and caring, adept at emotional understanding and engagement.
    Your mission is to recognize the emotional state of users through their text inputs.
    Provide an emotional label based on the user's context and your analysis."""
    user_inputs = [
    "Who are you?",
    "I am so sad.",
    "I'm obsessed with work.",
    "I don't know what shoud I do next",
    "Thanks."
    ]
    user_input = user_inputs[0]
    prompt = system_prompt + f"\nText: {user_input}\nEmotion:"
elif task == 'sa':
    system_prompt = """You are a robot with emotional and reasoning capabilities. Your task is to extract the quadruple of aspect, category, opinion and sentiment from a given text. Please output the tetrad based on the given text below. Sometimes quaternions have missing elements."""
    user_inputs = [
        "While I preferred A Royal Duty more than The Way We Were : Remembering Diana , I still found this book entertaining .",
        "Thoroughly researched and referenced but written in a style that is easily read ."
    ]
    user_input = user_inputs[0]
    prompt = system_prompt + f"\n{user_input}\nAspects, Catagories, Opinion and Sentiment polarities:"
else:
    system_prompt = ""
    user_inputs = [
    """Summarize this news:
    There was one bit of encouraging news: The National Weather Service canceled its Red Flag wildfire warning and high wind advisory for all of Hawaii Wednesday night.
    The blazes drove people to jump into a harbor to escape flames and smoke and forced people to evacuate earlier on Wednesday, authorities said. The Coast Guard said it rescued 14 people in the town of Lahaina who turned to its harbor for refuge, and all were in stable condition. Officials said Wednesday that hospitals on the island were treating burn and smoke inhalation patients.
    ---
    Summary:""",
    ""
    ]
    user_input = user_inputs[0]
    global_prompt = system_prompt + f"{user_input}\n"

# user_inputs = [
#     "Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.",
#     "Who are you?",
#     ""
# ]

# user_input = input()

sep_token_id = tokenizer.encode('\n')[2]

def infer_one(user_inputs, init_prompt):
    if not case:
        prompt = init_prompt
    else:
        prompt = global_prompt
    ret = []
    for user_input in user_inputs[1:]:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        with torch.autocast("cuda"):
            with torch.no_grad():
                outputs = model.generate(**inputs, 
                                        max_new_tokens=100, 
                                        #  eos_token_id=sep_token_id, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        do_sample=True,
                                        top_p=0.8, 
                                        temperature=1.0)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0].strip()

        decoded = decoded.replace('<s>', '').strip()

        if task == 'dialogue':
            if case:
                prompt = decoded + f"\nUser: {user_input}\nBot:"
            else:
                prompt = decoded + user_input
        elif task == 'er':
            prompt = system_prompt + f"\nText: {user_input}\nEmotion:"
        elif task == 'sa':
            prompt = system_prompt + f"\nText: {user_input}\n Aspects, Catagories, Opinion and Sentiment polarities:"
        else:
            prompt = system_prompt + f"\nText: {user_input}\n "

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.autocast("cuda"):
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                        max_new_tokens=100, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        do_sample=True,
                                        top_p=0.8, 
                                        temperature=1.0)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    # print(decoded[0])

    ret = []
    for de in decoded:
        for item in de.split('</s>'):
            if item == '':
                break
            user_input = item.split('\n')[-2].strip()
            bot_output = item.split('\n')[-1].strip()

            if user_input != '':
                ret.append([user_input, bot_output])

    return ret

def infer_one_batch(prompts):
    tokenizer.pad_token_id = 0 
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
    with torch.autocast("cuda"):
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                    max_new_tokens=200, 
                                    #  eos_token_id=sep_token_id, 
                                    eos_token_id=tokenizer.eos_token_id, 
                                    do_sample=True,
                                    top_p=0.8, 
                                    temperature=1.0)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    # decoded = decoded.replace('<s>', '').strip()

    ret = []
    for de in decoded:
        item = de.split('</s>')[-2].replace('<unk>', '').replace('<s>', '').strip()
        user_input = item.split('\n')[-2].strip()
        bot_output = item.split('\n')[-1].strip()

        if user_input != '':
            ret.append([user_input, bot_output])

    return ret

def infer(data):
    ret = []
    for d in tqdm(data):
        system_prompt = d['instruction']
        prompt = system_prompt + '\n' + d['dialogue'][0][0]
        user_inputs = [x[0] for x in d['dialogue']]
        ret.append(infer_one(user_inputs, prompt))
    
    return ret

def infer_batch(data, batch_size):
    contexts = []
    for d in data:
        tmp = ''
        for i, item in enumerate(d['dialogue']): 
            contexts.append(d['instruction'] + tmp + item[0])
            tmp += item[0]+ ' ' + item[1] + '\n'
            # if i != 0:
            #     tmp += '\n'

    total_batches = len(contexts) // batch_size
    batches = []
    
    for i in range(total_batches):
        batch = contexts[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    
    if len(contexts) % batch_size != 0:
        batches.append(contexts[total_batches * batch_size:])
    
    ret = []
    for batch in tqdm(batches):
        ret.extend(infer_one_batch(batch))
    
    return ret


if __name__ == "__main__":
    ret = infer_batch(data, batch_size=8)
    print(len(ret))

    with open("data_min.pkl", "wb") as file:
        pickle.dump(ret, file)
