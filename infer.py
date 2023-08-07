
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'
import argparse, logging

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
import json

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from tokenization import build_tokenized_dataset
from config import IGNORE_INDEX

use_lora = True
load_in_8bit = False
lora_hyperparams_file = 'config/lora.json'
model_name_or_path = '/data/jianggongyao/Llama-2-7b-hf/'
peft_path = '/data/jianggongyao/emola_output/checkpoint-46476'

config = LoraConfig.from_pretrained(peft_path)

tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit = load_in_8bit if use_lora else False,
    device_map='auto',
)
model.resize_token_embeddings(len(tokenizer))

if use_lora:
    model = prepare_model_for_int8_training(model)
    lora_hyperparams = json.load(open(lora_hyperparams_file))
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False, 
        r=lora_hyperparams['lora_r'], 
        lora_alpha=lora_hyperparams['lora_alpha'], 
        lora_dropout=lora_hyperparams['lora_dropout'], 
        bias=lora_hyperparams['lora_bias']
    )
    print(peft_config)
    model = get_peft_model(model, peft_config)

print(model)

system_prompt = """You are a digital companion with the ability to understand and empathize with a variety of human emotions.
Provide emotional comfort and companionship to users through meaningful dialogue.
Generate responses that affirm and validate the user's feelings, based on the context."""
user_inputs = [
    # "I am so sad.",
    "I'm obsessed with work.",
    "What should I do next?",
    "Thanks."
]
# user_input = input()
user_input = user_inputs[0]
prompt = system_prompt + f"\nUser: {user_input}\nBot: "
sep_token_id = tokenizer.encode('\n')[2]
# while True:
for user_input in user_inputs[1:]:
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=sep_token_id)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded)

    # user_input = input()
    prompt = decoded[0].strip() + f"\nUser: {user_input}.\n Bot: "