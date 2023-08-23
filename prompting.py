
import re
import json
import time


import openai
openai_config = json.load(open('config/openai.json'))
openai.api_base = openai_config['openai_api_base']
openai.api_key = openai_config ['api_key']


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


if __name__ == '__main__':
    # process_esconv('train')
    # summarize()
    pass