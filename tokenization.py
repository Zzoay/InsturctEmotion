
from itertools import chain

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

from prompt_wrap import *

model_name_or_path = '/data/jianggongyao/Llama-2-7b-hf/'
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
max_source_length = 640
max_target_length = 384
IGNORE_INDEX = -100


def tokenize_pair(data):
    ret = []

    # for sample in wrap_sentiment_triple():
    for sample in tqdm(data):
        source_ids = tokenizer.encode(text=sample['input_text'], add_special_tokens=True)
        target_ids = tokenizer.encode(text=sample['output_text'], add_special_tokens=False)

        if len(source_ids) > max_source_length:
            source_ids = source_ids[:max_source_length]
        if len(target_ids) > max_target_length - 1: # eos token
            target_ids = target_ids[:max_target_length - 1]

        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        labels = [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        # yield {
        #     'input_ids': input_ids,
        #     'attention_mask': [1] * len(input_ids),
        #     'labels': labels
        # }
        ret.append({
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels 
        })

    return ret

def tokenize_dialogue(data):
    ret = []
    for sample in tqdm(data):
        input_ids = tokenizer.encode(sample['instruction'], add_special_tokens=True)
        labels = [IGNORE_INDEX] * len(input_ids)
        for i, uttr in enumerate(sample['dialogue']):
            source_ids = tokenizer.encode(text=uttr[0], add_special_tokens=False)
            target_ids = tokenizer.encode(text=uttr[1], add_special_tokens=False)

            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length - 1: # eos token
                target_ids = target_ids[:max_target_length - 1]

            input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
        
        ret.append({
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels 
        })

    return ret

def build_tokenized_dataset(data_sources, sample_idx = -2):
    pair_data = []
    
    if data_sources.get("persona", False):
        pair_data.extend(wrap_persona_ext()[:sample_idx])
        pair_data.extend(wrap_peacok()[:sample_idx])
    
    if data_sources.get("commonsense", False):
        pair_data.extend(wrap_atomic()[:sample_idx])
        
    if data_sources.get("sa", False):
        pair_data.extend(wrap_aspect_sentiment_pair()[:sample_idx])
        pair_data.extend(wrap_sentiment_triple()[:sample_idx])
        pair_data.extend(wrap_sentiment_quadruple()[:sample_idx])
        
    if data_sources.get("er", False):
        pair_data.extend(wrap_emotion_recoginiton()[:sample_idx])
        
    ret = []
    ret.extend(tokenize_pair(pair_data))
    
    if data_sources.get("dialog", False):
        ret.extend(tokenize_dialogue(wrap_esc_like()[:sample_idx]))
    
    return ret


if __name__ == '__main__':
    # tokenize_pair()
    tokenize_dialogue(wrap_esc_like()[:500])
