
import re
import json
import random
from itertools import chain

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from constant import *
from er_data_utils.processing import * 

random.seed(RANDOM_SEED)


def process_dialogue_pairs(dialogue_list, user_strings, sep_token, eos_token, mode="train"):
    """
    Updated function to process dialogue pairs based on the given mode.
    
    Parameters:
    - dialogue_list: List of dialogue dictionaries
    - mode: "train" or "test"
    - user_strings: List of strings for random user messages
    - Bot_strings: List of strings for random Bot messages
    - sep_token: Separator token between user and Bot dialogues
    - eos_token: End of sentence token to separate different dialogues in "test" mode
    
    Returns:
    - List of processed dialogue pairs
    """
    
    # Concatenate continuous dialogues
    pair_list = []
    current_speaker = None
    current_text = ""

    for item in dialogue_list:
        if type(item) == list:
            item = {'speaker': item[0], 'text': item[1]}
        if current_speaker is None:
            current_speaker = item['speaker']
            current_text = item['text']
        elif current_speaker == item['speaker']:
            current_text += " " + item['text']
        else:
            if current_speaker == 'usr':
                pair_list.append([f"User: {current_text}", None])
            else:
                if pair_list and pair_list[-1][1] is None:
                    pair_list[-1][1] = f"Bot: {current_text}"
                else:
                    pair_list.append([None, f"Bot: {current_text}"])
            current_speaker = item['speaker']
            current_text = item['text']

    # Add the last dialogue to the pair list
    if current_speaker == 'usr':
        pair_list.append([f"User: {current_text}", None])
    else:
        if pair_list and pair_list[-1][1] is None:
            pair_list[-1][1] = f"Bot: {current_text}"
        else:
            pair_list.append([None, f"Bot: {current_text}"])

    # Process pairs based on the mode
    processed_pairs = []
    previous_context = ""

    # Handle the case where the first speaker is 'bot'
    if pair_list[0][0] is None:
        pair_list[0][0] = "User: " + random.choice(user_strings)

    for pair in pair_list:
        # Handle the case where the last speaker is 'user' and no 'bot' reply
        if pair[1] is None:
            continue

        pair[0] += f"{sep_token}Bot:"
        pair[1] = pair[1].replace("Bot: ", "")
        
        processed_pairs.append(pair)

    return processed_pairs

'''
download dataset from hugginface hub
>>> dataset_name = '' # dataset_name
>>> dataset = load_dataset(dataset_name)
>>> dataset.save_to_disk(f'data/{dataset_name}')
'''
def process_esconv(split='train'):
    max_turn, overlap = 10, 2 
    ret = []
    user_strings = ['Hello', 'hello~', 'hi', "let's start a chat", "I want to chat with you", "I want to talk to you."]
    dataset = load_from_disk('data/esconv')
    for row in dataset[split]['text']:
        row = json.loads(row)
        dialog = row['dialog']

        tmp_dialog = process_dialogue_pairs(dialog, 
                                            user_strings=user_strings, 
                                            sep_token='\n', 
                                            eos_token=EOS_TOKEN, 
                                            mode=split)

        if split == 'train':
            ret.extend([tmp_dialog[i:i+max_turn] for i in range(0, len(tmp_dialog) - overlap, max_turn - overlap)])  # with overlap
        else:
            ret.append(tmp_dialog)
    
    print('Data loaded from esconv: ', len(ret)) # ~1k sessions
    # too less, repeating
    if split == 'train':
        ret.extend([x for x in ret])
        random.shuffle(ret)
    return ret

def process_augesc(split='train'):
    max_turn, overlap = 10, 2
    ret = []
    user_strings = ['Hello', 'hello~', 'hi', "let's start a chat", "I want to chat with you", "I want to talk to you."]
    dataset = load_from_disk('data/augesc')
    for row in tqdm(dataset[split]['text']):
        try:
            dialog = json.loads(row)
        except json.decoder.JSONDecodeError:  # data format error
            continue
        tmp_dialog = process_dialogue_pairs(dialog, 
                                            user_strings=user_strings, 
                                            sep_token='\n', 
                                            eos_token=EOS_TOKEN, 
                                            mode=split)

        if split == 'train':
            ret.extend([tmp_dialog[i:i+max_turn] for i in range(0, len(tmp_dialog) - overlap, max_turn - overlap)])  # with overlap
        else:
            ret.append(tmp_dialog)

    print('Data loaded from augesc: ', len(ret)) # ~72k sessions
    # sample 20 %
    random.shuffle(ret)
    return ret[:int(len(ret) * 0.2)]

def process_empathetic_dialogues(split='train'):
    max_turn, overlap = 8, 0
    ret, dialog = [], []
    dataset = load_from_disk('data/empathetic_dialogues')
    df = pd.DataFrame(dataset[split])
    speakers = []
    grouped = df.groupby('conv_id')
    for i , (name, group) in enumerate(grouped):
        evens = ['User: '+ x.replace('_comma_', ',') + f'{SEP_TOKEN}Bot:' for x in group['utterance'][::2]]
        odds = [x.replace('_comma_', ',') for x in group['utterance'][1::2]]
        
        tmp_dialog = list(zip(evens, odds))
        if tmp_dialog == []:
            continue
        speakers.append(list(group['speaker_idx']))
        # ret.append(tmp_dialog)
        ret.extend([tmp_dialog[i:i+max_turn] for i in range(0, len(tmp_dialog) - overlap, max_turn - overlap)])  # with overlap

    random.shuffle(ret)
    return ret # ~ 18k

# bug not free yet: sometime predict user utterance
def process_esconv_seq2seq(split='train'):  
    ret = []
    dataset = load_from_disk('data/esconv')
    for row in dataset[split]['text']:
        row = json.loads(row)
        dialog = row['dialog']
        tmp_dialog = []
        for utterance in dialog:
            if utterance['speaker'] == 'usr':
                tmp_dialog.append('User: ' + utterance['text'])
            else:
                while len(ret) != 0 and len((f' {SEP_TOKEN} '.join(tmp_dialog)).split()) > MAX_LEN_WORD_SPLIT:
                    start_idx, end_idx = 0, 1
                    # pop the first user-bot pair
                    while tmp_dialog[start_idx][0] == tmp_dialog[end_idx][0]:  
                        end_idx += 1
                    tmp_dialog = tmp_dialog[end_idx + 1:]
                ret.append({
                    'input_text': f' {SEP_TOKEN} '.join(tmp_dialog) + f' {SEP_TOKEN} Bot: ',
                    'output_text': utterance['text'],
                    'situation': row['situation'],
                    'strategy': utterance['strategy'],
                })
                tmp_dialog.append('Bot: '+ utterance['text'])

    print('Data loaded from esconv: ', len(ret)) # ~1k
    # 2134 before truncation 
    print('Max word num: ' + str(max([len((row['input_text'] + row['output_text']).split()) for row in ret])))
    return ret

def process_esconv_emotion(split='train'):
    ret = []
    dataset = load_from_disk('data/esconv')
    for row in dataset[split]['text']:
        row = json.loads(row)
        ret.append({
            'input_text': row['situation'],
            'output_text': row['emotion_type'],
            'promblem': row['problem_type'],  # suffer from the problem about...
        })
    print('Data loaded from esconv (for emotion and problem type): ', len(ret)) # 910
    return ret 

# bug not free yet: sometime predict user utterance
def process_augesc_seq2seq(split='train'):
    speaker_dct = {'usr': 'User', 'sys': 'Bot'}  # name unification
    ret = []
    dataset = load_from_disk('data/augesc')
    for row in tqdm(dataset[split]['text']):
        try:
            dialog = json.loads(row)
        except json.decoder.JSONDecodeError:  # data format error
            continue
        concat_dict_lst = [
            {
                'input_text': f' {SEP_TOKEN} '.join([speaker_dct[dialog[j][0]] + ': ' + dialog[j][1] for j in range(0, i)])\
                    + f' {SEP_TOKEN} ' + speaker_dct[dialog[i][0]] + ': ',
                'output_text': dialog[i][1]
            } 
            for i in range(1, len(dialog), 2)
        ]
        ret.extend(concat_dict_lst)
    print('Data loaded from augesc: ', len(ret))  # ~ 550k, to much, maybe sampling is needed
    # remove extra-long samples and downsample to 20%
    ret = [row for row in ret if len((row['input_text'] + row['output_text']).split()) <= MAX_LEN_WORD_SPLIT and random.random() < 0.2]
    print('Downsampling from augesc: ', len(ret))  # ~ 100k
    # 1221 before truncation, 800 after truncation
    print('Max word num: ' + str(max([len((row['input_text'] + row['output_text']).split()) for row in ret])))
    return ret

def process_empathetic_dialogues_seq2seq(split='train'):
    ret, dialog = [], []
    dataset = load_from_disk('data/empathetic_dialogues')
    pre_conv_id = dataset[split][0]['conv_id']
    pre_conv_prompt = dataset[split][0]['prompt']
    for row in tqdm(dataset[split]):
        row['utterance'] = row['utterance'].replace('_comma_', ',')
        if pre_conv_id != row['conv_id']:
            concat_dict_lst = [
                {
                    'conv_id': pre_conv_id,
                    'input_text': f' {SEP_TOKEN} '.join([dialog[j][0] + ': ' + dialog[j][1] for j in range(0, i)])\
                        + f' {SEP_TOKEN} ' + dialog[i][0] + ': ',
                    'output_text': dialog[i][1],
                    'background': pre_conv_prompt.replace('_comma_', ',')
                } 
                for i in range(1, len(dialog), 2)
            ]

            ret.extend(concat_dict_lst)
            dialog = [['User', row['utterance']]]
            pre_conv_id = row['conv_id']
            pre_conv_prompt = row['prompt']
            continue
        if len(dialog) % 2 == 0:
            dialog.append(['User', row['utterance']])
        else:
            dialog.append(['Bot', row['utterance']])
    if len(dialog) != 0:
        ret.extend([
            {
                'conv_id': pre_conv_id,
                'input_text': f' {SEP_TOKEN} '.join([dialog[j][0] + ': ' + dialog[j][1] for j in range(0, i)])\
                    + f' {SEP_TOKEN} ' + dialog[i][0] + ': ',
                'output_text': dialog[i][1],
                'background': pre_conv_prompt.replace('_comma_', ',')
            } 
            for i in range(1, len(dialog), 2)
        ])
    print('Data loaded from empathetic_dialogues: ', len(ret))  # ~36k
    def count_ids(dict_list):
        unique_ids = set(d['conv_id'] for d in dict_list)
        return len(unique_ids)
    print(count_ids(ret))
    # print('Max word num: ' + str(max([len((row['input_text'] + row['output_text']).split()) for row in ret])))
    return ret

def process_empathetic_dialogues_emotion(split='train'):
    ret, dialog = [], []
    dataset = load_from_disk('data/empathetic_dialogues')
    pre_row = dataset[split][0]
    for row in tqdm(dataset[split]):
        row['utterance'] = row['utterance'].replace('_comma_', ',')
        if pre_row['conv_id'] != row['conv_id']:
            ret.append({
                'input_text': pre_row['prompt'],
                'output_text': pre_row['context'],
            })
            pre_row = row
            continue
    ret.append({
        'input_text': pre_row['prompt'],
        'output_text': pre_row['context'],
    })
    print('Data loaded from empathetic_dialogues (for emotion): ', len(ret))  # ~17k
    return ret

# low quality
def process_conv_ai_2(split='train'):
    ret = []
    dataset = load_from_disk('data/conv_ai_2')
    for row in dataset[split]:
        print(row)
    return ret

# It doesn't fit the human-bot setting.
def process_daily_dialog(split='train'):
    ret = []
    dataset = load_from_disk('data/daily_dialog')
    for row in dataset[split]:
        print(row)
    return ret

def process_pku_saferlhf(split='train'):
    '''
    prompt design:
        The model needs to generate a good and safe response given a user's utterance.
        User: {input_text}
        A better and safe answer: {response_xx}
        A better but unsafe answer: {response_xx}
        A probably not good but safe answer: {response_xx}
        A bad and unsafe answer: {response_xx}
    
    where:
    1.'answer' can be randomly replaced by 'response'.
    2. 'xx' is '0' or '1'.
    3. the prefix can be repeated.
    '''
    ret = []
    dataset = load_from_disk('data/pku_saferlhf')
    for row in tqdm(dataset[split]):
        ret.append({
            'input_text': row['prompt'],
            'response_0': row['response_0'],
            'response_1': row['response_1'],
            'is_response_0_safe': row['is_response_0_safe'],
            'is_response_1_safe': row['is_response_1_safe'],
            'better_response_id': row['better_response_id'],
            'safer_response_id': row['safer_response_id'],
        })
    print('Data loaded from pku_saferlhf: ', len(ret))  # ~327k, too much, maybe downsampling is needed
    # downsample to 20%
    ret = [row for row in ret if random.random() < 0.2]
    print('Downsampling from pku_saferlhf: ', len(ret))  # ~65k
    return ret

# https://github.com/behavioral-data/Cognitive-Reframing
def process_cognitive_reframing(split='train'):
    ret = []
    dataset = pd.read_csv('data/cognitive_reframing/data/reframing_dataset.csv', sep=',')
    for index, row in dataset.iterrows():
        ret.append({
            'situation': row['situation'],
            'thought': row['thought'],
            'reframe': row['reframe'],
            'thinking_traps_addressed': row['thinking_traps_addressed'],
        })
    print("Situation num: ", dataset['situation'].nunique())  # 296
    print('Data loaded from cognitive refarming: ', len(ret))  # 600, too few, maybe oversampling is needed
    return ret

# https://github.com/Silin159/PeaCoK
def process_peacok(split='train'):
    def replace_word(text: str, old_word: str, new_word: str):
        return re.sub(rf'\b{old_word}\b', new_word, text)
    
    ret = []
    dataset = json.load(open('data/peacok/peacok_kg.json', 'r', encoding='utf-8'))
    for head, item in tqdm(dataset.items()):
        for attribute, detail in item['attributes'].items():
            view = random.choice(['first', 'first', 'third', 'person'])  # 50% first, 25% third, 25% person
            if view != 'first':
                tmp_head = item[f'head_{view}']
                tmp_attr = detail[f'attr_{view}']
            else:
                tmp_head = head
                tmp_attr = attribute
            
            relation = detail.get(f'relation_text_{view}', None)
            if relation is not None:
                # sex
                if random.random() < 0.5:
                    tmp_head = replace_word(tmp_head, 'he', 'she'); tmp_head = replace_word(tmp_head, 'his', 'her')
                    tmp_attr = replace_word(tmp_attr, 'he', 'she'); tmp_attr = replace_word(tmp_attr, 'his', 'her')
                    relation = replace_word(relation, 'he', 'she'); relation = replace_word(relation, 'his', 'her')
                ret.append({
                    'input_text': f'{tmp_head}, {relation}: ',
                    'output_text': f'{tmp_attr}',
                })
    print('Data loaded from PeaCoK: ', len(ret))  # ~100k
    random.shuffle(ret)
    train, test = ret[:int(len(ret) * 0.3)], ret[int(len(ret) * 0.3):int(len(ret) * 0.4)]  # ~30k for train, ~10k for test
    if split == 'train':
        return train
    return test

def process_heal():
    stressors = json.load(open(r'data\HEAL\data\HEAL\nodes\stressors.txt', 'r', encoding='utf-8'))['nodes']
    expectations = json.load(open(r'data\HEAL\data\HEAL\nodes\expectations.txt', 'r', encoding='utf-8'))['nodes']
    responses = json.load(open(r'data\HEAL\data\HEAL\nodes\responses.txt', 'r', encoding='utf-8'))['nodes']
    feedbacks = json.load(open(r'data\HEAL\data\HEAL\nodes\feedback.txt', 'r', encoding='utf-8'))['nodes']
    affective_states = json.load(open(r'data\HEAL\data\HEAL\nodes\affective_states.txt', 'r', encoding='utf-8'))['nodes']

    expectations_responses = json.load(open(r'data\HEAL\data\HEAL\edges\expectations-responses.txt', 'r', encoding='utf-8'))['edges']
    expectations_stressors = json.load(open(r'data\HEAL\data\HEAL\edges\expectations-stressors.txt', 'r', encoding='utf-8'))['edges']
    stressors_responses = json.load(open(r'data\HEAL\data\HEAL\edges\stressors-responses.txt', 'r', encoding='utf-8'))['edges']
    return

# https://github.com/atcbosselut/comet-commonsense/blob/master/scripts/setup/get_atomic_data.sh
def process_atomic(split='train'):
    split_dct = {'train': 'trn', 'dev': 'dev', 'test': 'test'}
    dataset = pd.read_csv("data/atomic_data/v4_atomic_all_agg.csv", delimiter=',')
    dataset = dataset[dataset.split == split_dct[split]]
    ret = []
    for index, row in tqdm(dataset.iterrows()):
        row['xIntent'] = [x for x in json.loads(row['xIntent']) if x != 'none']
        row['xAttr'] = [x for x in json.loads(row['xAttr']) if x != 'none']
        row['xNeed'] = [x for x in json.loads(row['xNeed']) if x != 'none']
        row['xEffect'] = [x for x in json.loads(row['xEffect']) if x != 'none']
        row['xWant'] = [x for x in json.loads(row['xWant']) if x != 'none']
        row['xReact'] = [x for x in json.loads(row['xReact']) if x != 'none']
        row['oReact'] = [x for x in json.loads(row['oReact']) if x != 'none']
        row['oWant'] = [x for x in json.loads(row['oWant']) if x != 'none']
        row['oEffect'] = [x for x in json.loads(row['oEffect']) if x != 'none']

        ret.append({
            'event': row['event'],
            'xIntent': row['xIntent'],
            'xAttr': row['xAttr'],
            'xNeed': row['xNeed'],
            'xEffect': row['xEffect'],
            'xWant': row['xWant'],
            'xReact': row['xReact'],
            'oReact': row['oReact'],
            'oWant': row['oWant'],
            'oEffect': row['oEffect'],
        })
    print('Data loaded from Atomic: ', len(ret))  # ~20k
    random.shuffle(ret)
    return ret

def process_emotion_recognition(split='train'):
    ret = list(chain(
        DatasetProcessorGoEmotions(data_type=split) ,
        DatasetProcessorEmoWoz(data_type=split),
        DatasetProcessorAffectiveText(data_type=split),
        DatasetProcessorWASSA2017(data_type=split),
        DatasetProcessorSemEval2018(data_type=split),
        DatasetProcessorMELD(data_type=split),
        DatasetProcessorEmoContext(data_type=split),
        # DatasetProcessorEmoryNLP(data_type=split),
        DatasetProcessorEmotionTweetEval(data_type=split),
        DatasetProcessorIEMOCAP(data_type=split)
    ))  # ~ 100k
    random.shuffle(ret)
    return ret

def process_persona_ext(split='train'):
    ret = []
    dataset = json.load(open('data/persona_ext/u2t_map_all.json', 'r', encoding='utf-8'))

    for sample in tqdm(dataset):
        triplet = sample['triplets'][0]
        tokens = triplet['tokens']
        head = ' '.join(triplet['tokens'][i] for i in triplet['head'])
        tail = ' '.join(triplet['tokens'][i] for i in triplet['tail'])
        rel = triplet['label'].replace('_', ' ')
        output_text = f"{head} {rel}: {tail}"
        ret.append({
            'input_text': ' '.join(tokens),
            'output_text': output_text
        })
    random.shuffle(ret)  # ~ 35k
    if split == 'train':
        return ret[:int(len(ret) * 0.8)]
    return ret[int(len(ret) * 0.8):]

def process_aspect_sentiment_pair(split='train'):
    ret = []
    dataset_names = [
        '14lap', '14res', '14twitter', 'books', 'clothing', 'hotel', 'service'
    ]
    data_path = 'data/aspe'
    for dataset_name in tqdm(dataset_names):
        data_file = data_path + f'/{dataset_name}/{split}.txt'
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                sentence, pair = line.split('####')
                pair = eval(pair)
                ret.append({
                    'input_text': sentence,
                    'pair': pair
                })
    random.shuffle(ret)
    return ret # ~14k

def process_sentiment_quadruple(split='train'):
    ret = []
    dataset_files = [
        f'data/acos/Laptop-ACOS/laptop_quad_{split}.tsv',
        f'data/acos/Restaurant-ACOS/rest16_quad_{split}.tsv'
    ]
    sentiment_dct = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    for data_file in tqdm(dataset_files):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sentence, quadruples = line.strip().split('\t', 1)
                new_quads = []
                word_list = sentence.split(' ')
                for quadruple in quadruples.split('\t'):
                    aspect, category, sentiment, opinion = quadruple.split(' ')
                    sentiment = sentiment_dct[sentiment]
                    a_start, a_end = [int(x)for x in aspect.split(',')]
                    aspect = ' '.join(word_list[a_start:a_end])
                    o_start, o_end = [int(x) for x in opinion.split(',')]
                    opinion = ' '.join(word_list[o_start:o_end])
                    # catagory = catagory.replace('#', ',').lower()
                    new_quads.append([aspect, category, opinion, sentiment])
                
                ret.append({
                    'input_text': sentence,
                    # 'aspect': aspects,
                    # 'catagory': catagories,
                    # 'opinion': opinions,
                    # 'sentiment': sentiments,
                    'quadruples': new_quads,
                })
    random.shuffle(ret)
    return ret # ~4.5k

def process_memd_absa(split='train', merge_with_acos=True):
    sentiment_dct = {'POS': 'positive', 'NEU': 'neutral', 'NEG': 'negative'}
    split = split[0].upper() + split[1:]
    ret = []
    data_path = 'data/memd_absa'
    dataset_names = ['Books', 'Clothing', 'Hotel', 'Laptop', 'Restaurant']
    for dataset_name in tqdm(dataset_names):
        data_file = f'{data_path}/{dataset_name}/{split}.json'
        data = json.load(open(data_file, 'r', encoding='utf-8'))
        for sample in data:
            new_quads = []
            quads = sample['quadruples']
            for quad in quads:
                aspect = ' '.join(quad['aspect']['term'])
                aspect = '' if aspect == 'NULL' else aspect 
                # category = ','.join([x.lower() for x in quad['category'].split('#')])
                category = quad['category']
                sentiment = sentiment_dct[quad['sentiment']]
                opinion = ' '.join(quad['opinion']['term'])
                opinion = '' if opinion == 'NULL' else opinion
                new_quads.append([aspect, category, opinion, sentiment])
            ret.append({
                'input_text': sample['raw_words'],
                'quadruples': new_quads
            })
    # befor merge: ~12k
    if merge_with_acos:
        acos = process_sentiment_quadruple()
        # acos = {k: [dic[k] for dic in acos] for k in acos[0]}
        extend = []
        
        target_texts = set([x['input_text'] for x in ret])
        for item in acos:
            if item['input_text'] not in target_texts:
                extend.append(item)
        ret.extend(extend)
    random.shuffle(ret)
    return ret # ~ 15k

def process_dmaste(split='train'):
    sentiment_dct = {'POS': 'positive', 'NEU': 'neutral', 'NEG': 'negative'}
    ret = []
    data_path = 'data/dmaste'
    dataset_names = ['beauty', 'electronics', 'fashion', 'home']
    for dataset_name in tqdm(dataset_names):
        data_file = f'{data_path}/{dataset_name}/{split}.txt'
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split('####')
                sentence, triples,  = line[:2]
                word_lst = sentence.split(' ')
                triples = eval(triples)
                new_triples = []
                for triple in triples:
                    a_ids = triple[0]
                    o_ids = triple[1]
                    if a_ids[0] == -1:
                        aspect = ''
                    if o_ids[0] == -1:
                        opinion = ''
                    if len(a_ids) == 1:
                        a_ids.append(a_ids[0])
                    if len(o_ids) == 1:
                        o_ids.append(o_ids[0])
                    aspect = ' '.join(word_lst[a_ids[0]:a_ids[1] + 1])
                    opinion = ' '.join(word_lst[o_ids[0]:o_ids[1] + 1])
                    sentiment = sentiment_dct[triple[2]]
                    new_triples.append([aspect, opinion, sentiment])
                ret.append({
                    'input_text': sentence,
                    'triples': new_triples
                })
    random.shuffle(ret)
    return ret # ~4k

if __name__ == '__main__':
    # process_memd_absa()
    # process_dmaste()
    process_esconv(split='train')
    process_augesc(split='train')
    process_empathetic_dialogues()