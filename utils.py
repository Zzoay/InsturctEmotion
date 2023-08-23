
import random


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