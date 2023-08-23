
import os
os.environ['OPENAI_API_BASE'] = "https://api.openai-sb.com/v1"

import ast
import re
import json
import time
import random
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk

import openai
openai_config = json.load(open('config/openai.json'))
openai.api_base = openai_config['openai_api_base']
openai.api_key = openai_config ['api_key']


system_prompt = """Please list 100 common groups and 20 situations they may face each (not necessarily bad), which can lead to negative cognitions and thus induce negative thoughts. 

Note:
1. Please narrate in the first person, both the situation and negative thoughts should be complete sentences.
2. 20 situations for a group
3. 5 negative thoughts for a situation
"""

group_prompt = """
I will ask you to output the results in a step-by-step manner, you don't need to complete all the requirements at once. Firstly, please output 100 different groups, in python list format such as "[..., ...]" (output only contain the list).
"""

situation_prompt = """Please generate 20 situations (not necessarily bad or negative) that {group_name} may face, in the first person (for example, I...). Ensure JSON format (without extra information), where the keys are "group" and "situation" (ensure double quotes), like {{r"group": ..., r"situation": [{{r"situation_id": 1, r"situation": ...}}]}}. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_prompt = """Correspondingly, what negative thoughts might arise (due to cognitive traps) in response to these situations? It is in the first person (for example, I...). Keep the json format, as {"group": "...", "negative_thoughts": [{"situation_id": 1, "thoughts": [...,...]}, ...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

positive_prompt = """These negative thoughts may be due to cognitive traps. Please step out of these traps and turn them into positives (strict one-to-one). It is in the first person (for example, I...). Keep the json format, as {"situation_id": 1, "thoughts": [...,...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""


def get_response(history_messages, model="gpt-3.5-turbo-0613"):
    while True:
        try:
            completion = openai.ChatCompletion.create(model=model, temperature=1, messages=history_messages)
            print(completion.usage)
        except Exception as e:
            print(e)
            time.sleep(3)
            continue
        break
    return completion.choices[0].message.content.strip()

def get_groups(history_messages, load_from_file=False):
    history_messages.append({
        "role": "user",
        "content": group_prompt,
    })
    if load_from_file:
        with open('output/coc/groups.txt', 'r') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
        return history_messages, ret
    r = get_response(history_messages, model="gpt-4")
    # process to list
    list_string = r[r.find('['): r.rfind(']')+1]
    ret = ast.literal_eval(list_string)
    ret = json.loads(json.dumps(ret))
    with open('output/coc/groups.txt', 'a') as f:
        json.dump(ret, f)
    return history_messages, ret

def get_situations(history_messages, group_name, load_from_file=False):
    messages = history_messages.copy()
    messages.append({
        "role": "user",
        "content": situation_prompt.format(group_name=group_name)
    })
    if load_from_file:
        with open('output/coc/situations.txt', 'r') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
                if ret['group'] == group_name:
                    return messages, ret
    r = get_response(messages)  # TODO: double quotes
    list_string = r[r.find('{'): r.rfind('}')+1]
    ret = ast.literal_eval(list_string)
    # save to file
    with open(f'output/coc/situations.txt', 'a') as f:
        f.write('\n')
        json.dump(ret, f)
    # history_messages.append({
    #     "role": "assistant",
    #     "content": str(ret),
    # })
    return messages, ret

def get_negative_thoughts(history_messages, situations, load_from_file=False):
    messages = history_messages.copy()
    messages.append({
        "role": "assistant",
        "content": str(situations),
    })
    messages.append({
        "role": "user",
        "content": negative_prompt,
    })
    if load_from_file:
        with open('output/coc/negative_thoughts.txt', 'r') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
                if ret['group'] == situations['group']:
                    return messages, ret['negative_thoughts']
    r = get_response(messages)
    ret = ast.literal_eval(r)
    # save to file
    with open(f'output/coc/negative_thoughts.txt', 'a') as f:
        f.write('\n')
        json.dump(ret, f)
    # history_messages.append({
    #     "role": "assistant",
    #     "content": str(ret),
    # })
    return messages, []

def get_positive_thoughts(history_messages, situations, negative_thoughts, load_from_file=False):
    ret = {"group": situations['group'], "thoughts": []}
    for idx, negative_thought in enumerate(negative_thoughts):
        situation = situations['situations'][idx]
        messages = history_messages.copy()
        messages[-2] = {
            "role": "assistant",
            "content": "One of situations: " + str(situation),
        }
        messages.append({
            "role": "assistant",
            "content": str(negative_thought),
        })
        messages.append({
            "role": "user",
            "content": positive_prompt,
        })
        if load_from_file:
            with open('output/coc/positive_thoughts.txt', 'r') as f:
                for line in f.readlines():
                    ret = ast.literal_eval(line)
                    if ret['group'] == situations['group']:
                        return messages, ret['positive_thoughts']
        while True:
            try:
                r = get_response(messages)
                one_thought = ast.literal_eval(r)
                break
            except SyntaxError as e:
                print(e)
                continue
        ret["thoughts"].append(one_thought)
        messages = []
    
    with open(f'output/coc/positive_thoughts.txt', 'a') as f:
        f.write('\n')
        f.write(str(ret))
    return messages, ret

def chain_init_to_positive():
    history_messages = [{
        "role": "system",
        "content": system_prompt,
    }]
    history_messages, groups = get_groups(history_messages, load_from_file=True)
    for i, group_name in enumerate(groups):
        # if not os.path.exists(f'output/coc/{group_name}'):
        #     os.mkdir(f'output/coc/{group_name}')
        if i < 38:
            continue

        situation_msg, situations = get_situations(history_messages, group_name, load_from_file=True)
        situation_msg.pop(1)

        neg_msg, negative_thoughts = get_negative_thoughts(situation_msg, situations, load_from_file=True)
        pos_msg, positive_thoughts = get_positive_thoughts(neg_msg, situations, negative_thoughts, load_from_file=True)

if __name__ == "__main__":
    chain_init_to_positive()