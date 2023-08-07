
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,7'
import argparse, logging
import json

import fire
import random
import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers

from torch.utils.data import SequentialSampler
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


def set_all_seeds(seed_value):
    random.seed(seed_value)  
    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

set_all_seeds(42)

def train():
    model_name_or_path = '/data/jianggongyao/Llama-2-7b-hf/'
    use_lora = True
    lora_hyperparams_file = 'config/lora.json'
    config_file = 'config/config.json'
    deepspeed_config_file = 'config/deepspeed.json'
    output_dir = '/data/jianggongyao/emola_output/'
    load_in_8bit = True
    gradient_accumulation_steps = 1
    deepspeed_config = json.load(open(deepspeed_config_file))
    per_device_train_batch_size = deepspeed_config['train_micro_batch_size_per_gpu']

    model_config = json.load(open(config_file))

    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    train_data = Dataset.from_list(build_tokenized_dataset())
    print("train set size: " + str(len(train_data)))  # full: 247870

    val_set_size = model_config['val_set_size']
    val_data = None

    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    # tokenizer.add_special_tokens({"pad_token": "<PAD>",})
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit = load_in_8bit if use_lora else False,
        device_map=device_map,
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
        
    print("--start train--")
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=args.deepspeed,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=model_config["eval_steps"] if val_set_size > 0 else None,
            fp16=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_grad_norm=model_config['max_grad_norm'],
            num_train_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            load_best_model_at_end=False,
            logging_steps=model_config['logging_steps'],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            warmup_steps=model_config['warmup_steps'],
            save_strategy="epoch",
            save_steps=model_config["save_steps"],
            save_total_limit=3,
            report_to="wandb",
            # group_by_length=group_by_length
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            # label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
            label_pad_token_id=IGNORE_INDEX
        )
    )

    # not shuffle
    trainer._get_train_sampler = lambda: SequentialSampler(trainer.train_dataset)

    model.config.use_cache = False
    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
    with torch.autocast("cuda"):
        trainer.train()


if __name__ == '__main__':
    # train()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lora_hyperparams_file", default="", type=str, help="provide it when use_lora=True")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_lora", action="store_true", default=True, help="use lora")
    args = parser.parse_args()
    fire.Fire(train)

    print("------completed------")