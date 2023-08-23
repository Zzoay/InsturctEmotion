
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse, logging
import re

import fire
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from prompt_wrap import wrap_esc


use_lora = False
load_in_8bit = False
lora_hyperparams_file = 'config/lora.json'
model_name_or_path = '/data/jianggongyao/Llama-2-7b-chat-hf/'


tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left')
tokenizer.add_special_tokens({"pad_token":"<pad>"})

model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit = load_in_8bit,
    torch_dtype=torch.float16,
    use_safetensors=True,
    device_map='auto',
)
model.resize_token_embeddings(len(tokenizer))

model.eval()

# print(model)

# system_prompt = ""
system_prompt = """You are a digital companion with the ability to understand and empathize with a variety of human emotions.
Provide emotional comfort and companionship to users through meaningful dialogue.
Generate responses that affirm and validate the user's feelings, based on the context."""
# system_prompt = """You are a compassionate companion bot, understanding and caring, adept at emotional understanding and engagement.
# Your mission is to recognize the emotional state of users through their text inputs.
# Provide an emotional label based on the user's context and your analysis."""
# system_prompt = """You are a robot with emotional and reasoning capabilities. Your task is to extract the quadruple of aspect, category, opinion and sentiment from a given text. Please output the tetrad based on the given text below. Sometimes quaternions have missing elements.
# """
user_inputs = [
    "Who are you?",
    "I am so sad.",
    "I'm obsessed with work.",
    "I don't know what shoud I do next",
    "Thanks."
]

def infer_one(user_inputs):
    user_input = user_inputs[0]
    if user_input.strip() == '':
        user_input = 'hello'
    # llama2_tempalte = """"<s>[INST] <<SYS>>
    # {system_prompt}
    # <</SYS>>

    # {user_msg_1} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    # """
    llama2_tempalte = """"<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_msg_1} [/INST] """
    prompt = llama2_tempalte.format(system_prompt=system_prompt, user_msg_1=user_input)

    # while True:
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
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        decoded = decoded[0].replace("<s>", "").strip()
        # print(decoded + '\n')
        prompt = f"{decoded} </s><s>[INST] {user_input} [/INST]"
        # prompt = llama2_tempalte.format(system_prompt=system_prompt, user_msg_1=user_input)

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.autocast("cuda"):
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                        max_new_tokens=100, 
                                        # eos_token_id=sep_token_id, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        do_sample=True,
                                        top_p=0.8, 
                                        temperature=1.0)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    # print(decoded[0])

    ret = []
    for i, item in enumerate(re.split(r'\[INST\]|\[/INST\]', decoded[0])):
        if i == 0:  # <s>
            continue
        if i == 1:
            tmp = item.split('<</SYS>>')[-1].strip()
            continue
        
        if i % 2 != 0:
            tmp = item
        else:
            ret.append([tmp.replace('<s>', '').replace('</s>', '').strip(), item.replace('<s>', '').replace('</s>', '').strip()])
    
    return ret

def infer(data):
    ret = []
    for d in tqdm(data):
        ret.append(infer_one([x[0].replace('User: ', '').replace('Bot:', '').strip() for x in d['dialogue']]))
    return ret


def infer_one_batch(prompts):
    # tokenizer.pad_token_id = 2 
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
    with torch.autocast("cuda"):
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                    max_new_tokens=100, 
                                    #  eos_token_id=sep_token_id, 
                                    eos_token_id=tokenizer.eos_token_id, 
                                    do_sample=True,
                                    top_p=0.8, 
                                    temperature=1.0)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    # decoded = decoded.replace('<s>', '').strip()

    ret = []
    for de in decoded:
        de = de.split("<</SYS>>")[1].strip().replace("<unk>", "")
        part_1, part_2 = de.split("[/INST]")[-2], de.split("[/INST]")[-1]
        if "[INST]" in part_1:
            part_1 = part_1.split("[INST]")[1].strip()
        bot_output = part_2.replace("</s>", "").strip()
        user_input = part_1.strip()

        ret.append([user_input, bot_output])

    return ret


def infer_batch(data, batch_size):
    llama2_tempalte = """"<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_msg} [/INST] """
    contexts = []
    for d in data:
        tmp = ''
        for i, item in enumerate(d['dialogue']): 
            item[0] = item[0].replace('User: ', '').replace('Bot:', '').strip()
            item[1] = item[1].replace('User: ', '').replace('Bot: ', '').strip()
            # contexts.append(d['instruction'] + tmp + item[0])
            if i == 0:
                contexts.append(llama2_tempalte.format(system_prompt=d['instruction'], user_msg=item[0]))
                tmp += llama2_tempalte.format(system_prompt=d['instruction'], user_msg=item[0]) + item[1]
            else:
                contexts.append(tmp + " </s><s>[INST] {user_msg} [/INST]".format(user_msg=item[0]))
                tmp += " </s><s>[INST] {user_msg} [/INST] {bot_msg}".format(user_msg=item[0], bot_msg=item[1])
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


if __name__ == '__main__':
    data = wrap_esc(split='test')

    ret = infer_batch(data, batch_size=1)
    print(len(ret))

    import pickle
    with open("llama2_output_batch.pkl", "wb") as file:
        pickle.dump(ret, file)
