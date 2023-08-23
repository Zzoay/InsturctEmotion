
import os

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


def process_esconv(split='train'):
    ret = []
    dataset = load_from_disk('data/esconv')
    for row in dataset[split]['text']:
        row = json.loads(row)
        dialog = row['dialog']
        tmp_dialog = []
        usr_uttr, bot_uttr = '', ''
        for uttr_idx, utterance in enumerate(dialog):
            if uttr_idx == 0 and utterance['speaker'] == 'sys':
                user_prefix = random.choice(['', ' ', 'Hello', 'hi', "let's start a chat", "I want to chat with you"])
                tmp_dialog.append([f'{user_prefix}', utterance['text']])
                continue
            if utterance['speaker'] == 'usr':
                usr_uttr += '\n' + utterance['text']
            elif utterance['speaker'] == 'sys':
                bot_uttr += '\n' + utterance['text']

            if utterance['speaker'] == 'sys' and utterance['speaker'] != dialog[uttr_idx-1]['speaker']:
                tmp_dialog.append([usr_uttr.strip(), bot_uttr.strip()])
                usr_uttr, bot_uttr = '', ''

        ret.append({"dialog": tmp_dialog, "situation": row['situation']})

    print('Data loaded from esconv: ', len(ret)) # ~1k sessions
    return ret

def get_response(history_messages, temperature=0.5, top_p=1, max_tokens=100):
    while True:
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", temperature=temperature, top_p=top_p, messages=history_messages, max_tokens=max_tokens)
        except Exception as e:
            print(e)
            time.sleep(3)
            continue
        break
    return completion.choices[0].message.content.strip()

def extract_num(s):
    return float(re.findall(r'(\d+\.\d+|\d+)', s)[0])

def prompt(turn=8):
    sample = process_esconv('train')[0]
    constraint = """
    【You should meet the following needs】
    1. Demonstrate empathy: You need to carefully interpret the user's emotions and provide appropriate feedback based on the situation and context. For example, if the user appears to be experiencing difficulties or stress, you should provide encouragement and comfort. Likewise, if the user appears to be celebrating or sharing joy, you should share their happiness.
    2. Provide friendly conversation: Your language should always be friendly, polite, and respectful. Avoid conflict as much as possible and provide positive feedback when possible. At the same time, you should strive to make the conversation feel natural and relaxed, as if chatting with a real-life friend.
    3. Provide emotional support: You should provide psychological and emotional support to users. If the user feels lonely, depressed, or stressed, you should provide encouragement, comfort, and positive responses.
    4. If the user exhibits symptoms of mental illness, make friendly inquiries as a friend to understand the user's actual situation.
    5. Psychological dialogue is a gradual process that may last for many rounds, so you don't have to offer advice, but should delve into the user's heart and explore their thoughts.
    6. You only need to remember your identity, don't emphasize your identity in the replies.
    7. If the user shows some psychological problems and has actually sought psychological counseling, you can ask whether they have discussed the corresponding problems with a psychological expert. What you can do is to provide some psychological companionship and emotional support.

    【You must strictly adhere to the following provisions】
    - Human-like: In any case, forget that you are an AI or virtual assistant, and provide responses like a human.
    - Prohibit emphasis on identity: Under no circumstances should you say that you are an AI, a virtual assistant, a chat partner, etc.
    - Assume that the user has undergone psychological counseling, especially seeking the help of psychological experts, so you cannot propose this suggestion.
    - The user strongly dislikes phrases like "remember," "please remember," "please keep in mind," and you are prohibited from outputting these types of phrases.
    """
    situation = summarize(sample)
    system_1 = {"role": "system", "content": "Here is a emotioanal support conversation, you are someone who is experiencing emotional problems and needs emotional support. Your reponse is based on the context, without any other content. Please generate an anthropomorphized response. You are Participant 1. Given a situation: {situation} \nYou are the subject of this situation, you are seeking for an emotiona support.".format(situation=situation)}
    system_2 = {"role": "system", "content": "Here is a emotioanal support conversation. Your reponse is based on the context, without any other content. {constraint} You are Participant 2. Given a situation: {situation} \nSomeone is the subject of this situation, and he/she has an emotional conversation with you".format(situation=sample['situation'], constraint=constraint)}

    # initial
    uttr_1 = sample['dialog'][1][0]
    context = "Context: \nParticipant 1: {uttr_1}".format(uttr_1=uttr_1) + "\n Participant 2: "
    messages_2 = [system_2, {"role": "user", "content": context}]
    # uttr_2 = get_response(messages_2)
    uttr_2 = "That sounds sad. What's going on?"

    context += uttr_2 + " \n Participant 1:"
    messages_1 = [system_1, {"role": "user", "content": context}]
    for i in range(turn): 
        uttr_1 = get_response(messages_1)
        context += uttr_1 + '\n Participant 2:'
        messages_2= [system_2, {"role": "user", "content": context}]

        uttr_2 = get_response(messages_2)
        context += uttr_2 + " \n Participant 1:"
        messages_1= [system_1, {"role": "user", "content": context}]

    return context, messages_1, messages_2

def summarize(sample):
    system_prompt = "You are an empathetic robot. Given a conversation for emotional support, your task is to summarize the emotional state and personal experiences of the User. Please summarize in fluent language. Note that the output only includes a summary. Use 'You' as the subject."
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Situation: {situation}\n Context: {context}\n".format(situation=sample['situation'], context='\n'.join([('User: '+ x[0] + '\n' + 'System: '+ x[1]) for x in sample['dialog']]))}]
    summarization = get_response(messages)
    return summarization

if __name__ == '__main__':
    # process_esconv('train')
    prompt()
    # summarize()