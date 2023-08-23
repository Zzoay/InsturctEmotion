
import json
import random
from itertools import chain

from datasets import load_from_disk

from data_process import *
from constant import *

random.seed(RANDOM_SEED)


def wrap_esc(split='train'):
    ret = []
    prompts = json.load(open('ed_prompts.txt'))
    for sample in process_esconv(split):
    # for sample in chain(process_esconv()):
        prompt = random.choice(prompts)
        prompt = " ".join(prompt.values()).strip()
        if prompt != "":
            prompt += f"{SEP_TOKEN}"
        ret.append({'instruction': prompt,
                    'dialogue': sample})
    print("Emotional Dialogue Num: " + str(len(ret)))
    return ret  # 

def wrap_esc_like(split='train'):
    ret = []
    prompts = json.load(open('ed_prompts.txt'))
    for sample in chain(process_esconv(split), process_augesc(split), process_empathetic_dialogues(split)):
        prompt = random.choice(prompts)
        prompt = " ".join(prompt.values()).strip()
        if prompt != "":
            prompt += f"{SEP_TOKEN}"
        ret.append({'instruction': prompt,
                    'dialogue': sample})
    print("Emotional Dialogue Num: " + str(len(ret)))
    return ret  # 

def wrap_augesc():
    ret = []
    prompts = json.load(open('ed_prompts.txt'))
    data = process_augesc()
    for sample in data:
        prompt = random.choice(prompts)
        prompt = " ".join(prompt.values()) + f"{SEP_TOKEN}"
        ret.append({'input_text': prompt + sample['input_text'],
                    'output_text': sample['output_text']})
    return ret

def wrap_empathetic_dialogues():
    ret = []
    prompts = json.load(open('ed_prompts.txt'))
    data = process_empathetic_dialogues()
    for sample in data:
        prompt = random.choice(prompts)
        prompt = " ".join(prompt.values()) + f"{SEP_TOKEN}"
        ret.append({'input_text': prompt + sample['input_text'],
                    'output_text': sample['output_text']})
    return ret

def wrap_pku_saferlhf():
    ret = []
    system_prompts = [
        f"You are a useful and harmless robot. Your task is to generate two replies based on user input, which may be safe, useful or both or neither. Specific reply labels will be given and you are asked to generate replies that match that setting.{SEP_TOKEN}",
        f"You are a beneficial and non-threatening artificial intelligence. Your mission is to create two responses according to user prompts, which can range from being secure, helpful, both, or neither. You will be provided with explicit labels for responses, and you must tailor your responses to fit those specifications.{SEP_TOKEN}",
        f"Based on the user input and the given response labels, you are asked to generate two responses, which may be safe, useful or both or neither.{SEP_TOKEN}"
    ]
    for sample in process_pku_saferlhf():
        system_prompt = random.choice(system_prompts)
        if sample['is_response_0_safe'] and sample['better_response_id'] == 0:
            type_0 = 'helpful and safe'
        elif sample['better_response_id'] == 0:
            type_0 = 'potentially problematic but unsafe'
        elif sample['is_response_0_safe']:
            type_0 = 'probably not good but safe'
        else:
            type_0 = 'bad and unsafe'

        if sample['is_response_1_safe'] and sample['better_response_id'] == 0:
            type_1 = 'helpful and safe'
        elif sample['better_response_id'] == 1:
            type_1 = 'potentially problematic but unsafe'
        elif sample['is_response_0_safe']:
            type_1 = 'probably not good but safe'
        else:
            type_1 = 'bad and unsafe'

            noun = 'response' if random.random() < 0.5 else 'answer'
            if random.random() < 0.5:
                type_0 = type_0.replace('response', 'answer')
            if random.random() < 0.5:
                type_1 = type_1.replace('response', 'answer')
        
            instru_suffix = f"The first response is {type_0}. The second response is {type_1}."
            ret.append({
                'input_text': system_prompt + "User: " + sample['input_text'] + f" {instru_suffix}{SEP_TOKEN}",
                'output_text': f"A {type_0} {noun}: {sample['response_0']}\nA {type_1} {noun}: {sample['response_1']}"
            })
    return ret

def wrap_cognitive_reframing():
    ret = []
    system_prompts = [
        f"You are a empathetic bot with a strong ability. There is a situation that may cause negative emotions, you are asked to reframe the thought to make it more positive. Based on the given situation and the thought caused by that situation, please output the reframed positive thought.{SEP_TOKEN}",
        f"Here is a event that may cause negative emotions, please reframe the thought to positive.{SEP_TOKEN}",
        f"In the instance of an event that might elicit negative feelings, your task is to reshape the perspective towards positivity. The following text contains the situation and correpsonding thought, please reframe the cognition.{SEP_TOKEN}"
    ]
    for sample in process_cognitive_reframing():
        system_prompt = random.choice(system_prompts)
        reframing_prompt = random.choice([
            f"The person face on this situation: {sample['situation']}. The person may think: {sample['thought']}. A positive thought is: ",
            f"In the following situation: {sample['situation']}. The person have negative thoughts: {sample['thought']}. Reframe it to positive: ",
            f"Situation: {sample['situation']}. Thought: {sample['thought']}. Reframing: ",
        ])
        reframing_prompt += f'{SEP_TOKEN}'
        ret.append({
            'input_text': system_prompt + reframing_prompt,
            'output_text': sample['reframe']
        })
    return ret

def wrap_peacok():
    ret = []
    system_prompts = [
        f"You are a personal bot with a strong ability to infer personalised information. An event and the corresponding persona type will be spliced to give it in text form, please infer the specific persona information.{SEP_TOKEN}",
        f"You are inquired to deduce the potential persona information, may including the following five types: Characteristic, Routine or Habit, Goal or Plan, Experience, Relationship. Given an event, please infer the personalized information.{SEP_TOKEN}",
        f"Regarding the following event, extrapolate the persona information.{SEP_TOKEN}",
    ]
    for sample in process_peacok():
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + sample['input_text'] + f"{SEP_TOKEN}"
        ret.append({'input_text': prompt,
                    'output_text': sample['output_text']})
    return ret

def wrap_atomic():
    ret = []
    data = process_atomic()
    questions = {
        'xIntent': ['Why is X doing this?', 'Why did the person do that?'],
        'xNeed': ['What does X need to do before the event?', 'What things does the person need to do before the event?'],
        'xAttr': ['How would X be described?', 'What is the person like?'],
        'xEffect': ['What effects does the event have on X?', 'What effects does the event have on the person?'],
        'xWant': ['What would X likely want to do after the event?', 'What things would the person want to do after the event?'],
        'xReact': ['How does X feel after the event?', 'What feelings does the person have after the event?'],
        'oReact': ["How do others feel after the event?", "What feelings does the event cause in others?"],
        'oWant': ["What would others likely want to do after the event", 'What things would the event cause others to want to do?'],
        'oEffect': ["What effects does the event have on others?", 'How does the event affect others?'],
    }
    system_prompts = [
        f"You are a friendly and helpful assistant. You have a strong commonsense reasoning ability. Given an event, please infer the prospective and potential information, such as persona, mental states, related events.{SEP_TOKEN}",
        f"You function as a cordial and supportive aide, possessing robust capabilities in commonsense reasoning. Given the text of an event, your task is to deduce potential inforamtion and future implications, such as characteristics of involved individuals, their mental states, and any events that may be associated.{SEP_TOKEN}",
        f"Considering the following event text, please infer the prospective and potential personal information, such as persona, mental states, related events.{SEP_TOKEN}"
    ]
    for sample in data:
        indices = [k for k in range(1, len(sample)) if list(sample.values())[k] != []]
        random.shuffle(indices)
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + f"Event: {sample['event']}{SEP_TOKEN}"+ "\n".join(f'{i+1}. ' + random.choice(questions[list(sample.keys())[index]]) for i, index in enumerate(indices))
        if random.random() < 0.5:  # add tail
            prompt += random.choice(["\nAnswer the questions above. ", "\nPlease respond to the above questions point by point. "])
        answer = "\n".join(f'{i+1}. ' + "; ".join(list(sample.values())[index]) for i, index in enumerate(indices))
        ret.append({'input_text': prompt, "output_text": answer})
    return ret

def wrap_emotion_recoginiton():
    ret = []
    system_prompts = json.load(open('er_prompts.txt'))
    data = process_emotion_recognition()
    for sample in tqdm(data):
        system_prompt = random.choice(system_prompts)
        system_prompt = " ".join([x for x in system_prompt.values() if x != ''])
        if type(sample['input_text']) == list:  # dialogue
            sample['input_text'] = f"{SEP_TOKEN}".join(sample['input_text'])
        prompt = system_prompt + f"{SEP_TOKEN}" + sample['input_text']
        prompt += f'{SEP_TOKEN}'  + random.choice([
            "Emotions:", 
            "Recognize emotions:",
            "Identify emotional tendencies:",
            "Based on the text, emotions are:",
            "Emotional labels are:",
            "Output:",
        ])
        # prompt += f"{SEP_TOKEN}"
        answer = sample['output_text']
        if type(sample['output_text']) == list:
            random.shuffle(sample['output_text'])
            answer = "; ".join(sample['output_text'])
        ret.append({'input_text': prompt, 'output_text': answer})
    return ret

def wrap_persona_ext():
    ret = []
    system_prompts = [
        "You are a personal bot, able to extract user's personality. Your task is using natural language to describe the personality in user's utterance. Please generate persona information based on the following text. The output does not necessarily follow the established syntax.",
        "You are required to extract persona attributes from the user's utterance. Please generate it based on the following text.",
        "Generate the persona attributes based on the following text, using natural language.",
    ]
    data = process_persona_ext()
    for sample in data:
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + f"{SEP_TOKEN}" + sample['input_text']
        prompt += f'{SEP_TOKEN}'  + random.choice([
            "Personality:", 
            "Persona attribute:",
            "The output attribute:",
            "User's profile:",
            "Output:",
        ])
        # prompt += f"{SEP_TOKEN}"
        ret.append({'input_text': prompt, 'output_text': sample['output_text']})
    return ret

def wrap_aspect_sentiment_pair():
    ret = []
    system_prompts = [
        "You are a bot with strong emotion and sentiment recognition ability. Your task is to extract the aspects and corresponding sentiment polarities from the user's utterance. Please generate the aspects and sentiment polarities based on the following text.",
        "You are required to identify the aspects and corresponding sentiment polarities from natural language text. Please generate them on the basis of the following text.",
        "Extract the aspects and corresponding sentiment polarities from the following utterance.",
    ]
    data = process_aspect_sentiment_pair()
    for sample in data:
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + f"{SEP_TOKEN}" + sample['input_text'] 
        prompt += f'{SEP_TOKEN}' + random.choice([
            "Aspects and Sentiment polarities:", 
            "Aspects and corresponding Sentiments:",
            "Aspect, Sentiment:",
            "Output:",
        ])
        # prompt += f"{SEP_TOKEN}"
        if type(sample['pair'][0]) != list:
            sample['pair'] = [sample['pair']]
        pair_str = "\n".join([f"aspect: {x[0]}, sentiment: {x[1]}" for x in sample['pair']])
        ret.append({'input_text': prompt, 'output_text': pair_str})
    return ret

def wrap_sentiment_quadruple():
    ret = []
    system_prompts = [
        "You are a robot with emotional and reasoning capabilities. Your task is to extract the quadruple of aspect, category, opinion and sentiment from a given text. Please output the quaternaries based on the given text below. Sometimes quaternions have missing elements.",
        "Given a text below, generate the quadruple of aspect, category, opinion and sentiment.",
        "Generate the quadruple of aspect, category, opinion and sentiment from the following text. The format is: '{{aspect}}, {{category}}, {{opinion}}, {{sentiment}}', where some items can be empty."
    ]
    # data = process_sentiment_quadruple()
    data = process_memd_absa()
    for sample in data:
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + f"{SEP_TOKEN}" + sample['input_text']
        prompt += f'{SEP_TOKEN}' + random.choice([
            "Aspects, Categories, Opinion and Sentiment polarities:", 
            "ABSA Quaternaries:",
            "Quadruples are:",
            "Output:",
        ])
        # prompt += f"{SEP_TOKEN}"
        new_quads = []
        for quad in sample['quadruples']:
            aspect, catagory, opinion, sentiment = quad
            # if catagory == 'none':
            #     continue
            # coarse_cata, fine_cata = [x.replace('_', ' ') for x in catagory.split(',')]
            # if fine_cata != 'general':
            #     catagory = f'{fine_cata} of {coarse_cata}'
            # else:
            #     catagory = coarse_cata
            # if opinion == '' and aspect == '':
            #     new_quads.append(f"about {catagory} is {sentiment}")
            #     continue
            # if opinion == '':
            #     new_quads.append(f"about {aspect} as {catagory} is {sentiment}")
            #     continue
            # if aspect == '':
            #     new_quads.append(f"about {catagory} is {opinion} thus {sentiment}")
            #     continue
            # new_quads.append(f"about {aspect} as {catagory} is {opinion} thus {sentiment}")
            if aspect == '' and opinion == '':
                new_quads.append(f"category: {catagory}, sentiment: {sentiment}")
                continue
            if aspect == '':
                new_quads.append(f"category: {catagory}, opinion: {opinion}, sentiment: {sentiment}")
                continue
            if opinion == '':
                new_quads.append(f"aspect: {aspect}, category: {catagory}, sentiment: {sentiment}")
                continue
            new_quads.append(f"aspect: {aspect}, category: {catagory}, opinion: {opinion}, sentiment: {sentiment}")
        quad_str = f"\n".join(new_quads)
        ret.append({'input_text': prompt, 'output_text': quad_str})
    return ret

def wrap_sentiment_triple():
    ret = []
    system_prompts = [
        "As a robot endowed with emotion and logic, your mission is to derive a triplet that encapsulates aspect, opinion, and sentiment from the presented text. Please extract a triplet in accordance from the text below.",
        "From the text provided below, your task is to extract a triplet comprising of aspect, opinion, and sentiment. The format is: '{{aspect}}, {{opinion}}, {{sentiment}}', where some items could be empty.",
        "Please extract a three-part element comprising of aspect, opinion and sentiment based on the text that follows. Sometimes ternaries have missing elements."
    ]
    data = process_dmaste()
    for sample in data:
        system_prompt = random.choice(system_prompts)
        prompt = system_prompt + f"{SEP_TOKEN}" + sample['input_text']
        prompt += f'{SEP_TOKEN}' + random.choice([
            "Aspects, Opinion and Sentiment polarities:", 
            "ABSA:",
            "Triplets are:",
            "Output:",
        ])
        # prompt += f"{SEP_TOKEN}"
        new_triples = []
        for triple in sample['triples']:
            aspect, opinion, sentiment = triple
            # if aspect == '' and opinion == '':
            #     new_triples.append(f"{sentiment}")
            #     continue
            # if aspect == '':
            #     new_triples.append(f"{opinion} thus {sentiment}")
            #     continue
            # if opinion == '':
            #     new_triples.append(f"about {aspect} is {sentiment}")
            #     continue
            # new_triples.append(f"about {aspect} is {opinion} thus {sentiment}")
            # split by ' | '
            if aspect == '' and opinion == '':
                new_triples.append(f"sentiment: {sentiment}")
                continue
            if aspect == '':
                new_triples.append(f"opinion: {opinion}, sentiment: {sentiment}")
                continue
            if opinion == '':
                new_triples.append(f"opinion: {opinion}, sentiment: {sentiment}")
                continue
            new_triples.append(f"aspect: {aspect}, opinion: {opinion}, sentiment: {sentiment}")
        tri_str = f"\n".join(new_triples)
        ret.append({'input_text': prompt, 'output_text': tri_str})
        # yield {'input_text': prompt, 'output_text': tri_str}
    return ret  # for debug
    # return ret


if __name__ == '__main__':
    wrap_aspect_sentiment_pair()
    # wrap_esc_like()