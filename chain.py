
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
api_key = 'sb-c1846034d6d59562cfb8124f7b383d280db3f249c58bda1d'
openai.api_key = api_key


system_prompt = """Please list 100 common groups and 20 situations they may face each (not necessarily bad), which can lead to negative cognitions and thus induce negative thoughts. 

Note:
1. Please narrate in the first person, both the situation and negative thoughts should be complete sentences.
2. 20 situations for a group
3. 5 negative thoughts for a situation
"""

group_prompt = """
I will ask you to output the results in a step-by-step manner, you don't need to complete all the requirements at once. Firstly, please output 100 different groups, in python list format such as "[..., ...]" (output only contain the list).
"""

situation_prompt = """Please generate 20 situations (not necessarily bad or negative) that {group_name} may face, in the first person. Ensure JSON format (without extra information), where the keys are "group" and "situation" (ensure double quotes), like {{r"group": ..., r"situation": [{{r"situation_id": 1, r"situation": ...}}]}}. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_prompt = """Correspondingly, what negative thoughts might arise in response to these situations? Keep the json format, keys are "negative_thoughts". Don't output "situation" for clarity.
"""

positive_prompt = """These negative thoughts may be due to cognitive traps. Please step out of these traps and turn them into positives. Keep the JSON format, keys are "reframed_thoughts". Don't output "situation" and "negative_thoughts" for clarity.
"""


def get_response(history_messages, model="gpt-3.5-turbo-0613"):
    while True:
        try:
            completion = openai.ChatCompletion.create(model=model, temperature=0.2, messages=history_messages)
        except openai.error.RateLimitError or openai.error.ServiceUnavailableError or openai.error.APIConnectionError:
            time.sleep(10)
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

def get_situations(history_messages, group_name):
    history_messages.append({
        "role": "user",
        "content": situation_prompt.format(group_name=group_name)
    })
    r = get_response(history_messages)  # TODO: double quotes
    list_string = r[r.find('{'): r.rfind('}')+1]
    ret = ast.literal_eval(list_string)
    # save to file
    with open(f'output/coc/situations.txt', 'a') as f:
        json.dump(ret, f)
    history_messages.append({
        "role": "assistant",
        "content": str(ret),
    })
    return history_messages, ret

def get_negative_thoughts(history_messages):
    history_messages.append({
        "role": "user",
        "content": negative_prompt,
    })
    r = get_response(history_messages)
    list_string = r[r.find('{'): r.rfind('}')+1]
    ret = ast.literal_eval(list_string)
    # save to file
    with open(f'output/coc/negative_thoughts.txt', 'a') as f:
        json.dump(ret, f)
    history_messages.append({
        "role": "assistant",
        "content": str(ret),
    })
    return history_messages, ret

def get_positive_thoughts(history_messages):
    history_messages.append({
        "role": "user",
        "content": positive_prompt,
    })
    r = get_response(history_messages)
    list_string = r[r.find('{'): r.rfind('}')+1]
    ret = ast.literal_eval(list_string)
    # save to file
    with open(f'output/coc/positive_thoughts.txt', 'a') as f:
        f.write(str(ret))
    history_messages.append({
        "role": "assistant",
        "content": str(ret),
    })
    return history_messages, ret

def chain_init_to_positive():
    history_messages = [{
        "role": "system",
        "content": system_prompt,
    }]
    history_messages, groups = get_groups(history_messages, load_from_file=True)
    for group_name in groups:
        # if not os.path.exists(f'output/coc/{group_name}'):
        #     os.mkdir(f'output/coc/{group_name}')
        history_messages, situations = get_situations(history_messages, group_name)
        history_messages, negative_thoughts = get_negative_thoughts(history_messages)
        history_messages, positive_thoughts = get_positive_thoughts(history_messages)

if __name__ == "__main__":
    chain_init_to_positive()