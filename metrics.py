
import time
import random
import multiprocessing

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from prompting import get_response, extract_num
from data_process import process_esconv


def calculate_metrics(hypothesis, reference):
    ref = reference.split()
    hyp = hypothesis.split()

    # BLEU with smoothing
    smoothing = SmoothingFunction().method1
    bleu_1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference)[0]

    # ACCURACY
    if len(hyp) < len(ref):
        hyp += ['***'] * (len(ref) - len(hyp))
    accuracy = accuracy_score(ref, hyp[:len(ref)])

    # Distinct-1 and Distinct-2
    def distinct_n(sentence, n):
        words = sentence.split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        return len(ngrams) / (len(words) - n + 1)

    distinct_1 = distinct_n(hypothesis, 1)
    distinct_2 = distinct_n(hypothesis, 2)

    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-4": bleu_4,
        "ROUGE-1": rouge_scores['rouge-1']['f'],
        "ROUGE-2": rouge_scores['rouge-2']['f'],
        "ROUGE-L": rouge_scores['rouge-l']['f'],
        "ACCURACY": accuracy,
        "Distinct-1": distinct_1,
        "Distinct-2": distinct_2
    }

def calculate_metrics_avg(hyps, refs):
    scores = [calculate_metrics(h, r) for h, r in zip(hyps, refs)]
    avg_scores = {key: sum([score[key] for score in scores]) / len(scores) for key in scores[0]}
    return avg_scores

def eval_dialog_by_gpt(hyp_dialog, ref_dialog):
    # utterance-level
    win_rate_utterance = 0.0
    match_ref_score = 0.0
    coherence_score = 0.0
    informativeness_score = 0.0

    # dialogue-level
    win_rate_dialog = 0.0
    empathy_score = 0.0

    hyp_dialog_str, ref_dialog_str = '', ''
    num_turns = len(ref_dialog)     
    for hyp, ref in zip(hyp_dialog, ref_dialog):
        _, bot_uttr_hyp = hyp
        user_uttr, bot_uttr_ref = ref
        pair = [bot_uttr_hyp, bot_uttr_ref]
        ids = [0, 1]
        random.shuffle(ids) # random exchange
        win_p = f"Response 0:{pair[ids[0]]}\nReference Response 1:{pair[ids[1]]}"
        # win_utterance = get_response(
        #     [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Below, I will give you a user input and two replies. Please choose the better one."}, 
        #      {"role": "user", "content": f"User:{user_uttr}\n{win_p}\nWhich one is better? Your response is chosed from only the ['0', '1'], without any other contents."}]
        #     )
        # if str([pair[idx] for idx in ids].index(bot_uttr_hyp)) in win_utterance.lower():
        #     win_rate_utterance += 1 / num_turns
        
        match_p = random.choice([f"Hypothesis Response 1:{bot_uttr_hyp}\nReference 2:{bot_uttr_ref}", f"Reference Response:{bot_uttr_ref}\nHypothesis Response:{bot_uttr_hyp}"])
        # match_ref_score += extract_num(get_response(
        #     [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Given the Hypothesis and Reference, please analyze the degree to which they match. Score is from 1 to 5."}, 
        #      {"role": "user", "content": f"User:{user_uttr}\n{match_p}\nScore:"}]
        #     )) / num_turns
        # coherence_score += extract_num(get_response(
        #     [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Please score the response from the perspective of Coherence. Score is from 1 to 5."}, 
        #      {"role": "user", "content": f"User:{user_uttr}\nResponse:{bot_uttr_hyp}\nCoherence Score:"}]
        #     )) / num_turns
        # informativeness_score += extract_num(get_response(
        #     [{"role": "system", "content": "You are an NLP expert, good at judging responses in emotional dialogues. Please score the response from the perspective of Informativeness. Score is from 1 to 5."}, 
        #      {"role": "user", "content": f"User:{user_uttr}\nResponse:{bot_uttr_hyp}\nInformativeness Score:"}]
        #     )) / num_turns
        
        if bot_uttr_hyp.startswith('Bot'):
            hyp_dialog_str += f"{user_uttr}\n{bot_uttr_hyp}\n"
            ref_dialog_str += f"{user_uttr}\n{bot_uttr_ref}\n"
        else:
            hyp_dialog_str += f"{user_uttr}\nBot:{bot_uttr_hyp}\n"
            ref_dialog_str += f"{user_uttr}\n{bot_uttr_ref}\n"
    win_rate_dialog = 0.0
    pair = [hyp_dialog_str, ref_dialog_str]
    ids = [0, 1]
    random.shuffle(ids) # random exchange 
    win_p = f"Response 0:{pair[ids[0]]}\nResponse 1:{pair[ids[1]]}"
    # win_dialog = get_response(
    #         [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Below, I will give you a user input and two replies. Please choose the better one."}, 
    #          {"role": "user", "content": f"User:{user_uttr}\n{win_p}\nWhich one is better? Your response is chosed from only the ['0', '1'], without any other contents."}]
    # ) 
    # if str([pair[idx] for idx in ids].index(hyp_dialog_str)) in win_dialog.lower():
    #     win_rate_dialog = 1.0

    # match_ref_score = extract_num(get_response(
    #     [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Given the Hypothesis dialogs and Reference dialogs, please analyze the degree to which they match. Ignore the user's input, and focus on bot's responses. Don't have a preference for long text. Score is from 1.0 to 5.0."}, 
    #         {"role": "user", "content": f"Referecne: {ref_dialog_str}\nHypothesis: {hyp_dialog_str}\n Matching Score:"}],
    #     temperature = 0.2,
    #     max_tokens = 20,
    #     ))
    # coherence_score = extract_num(get_response(
    #     [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Please score the bot's responses from the perspective of Coherence. Don't have a preference for long text. Score is from 1.0 to 5.0."}, 
    #         {"role": "user", "content": f"Dialog: {hyp_dialog_str}\nCoherence Score:"}],
    #     temperature = 0.2,
    #     max_tokens = 20,
    #     ))
    # informativeness_score = extract_num(get_response(
    #     [{"role": "system", "content": "You are an NLP expert, good at judging responses in emotional dialogues. Please score the bot's responses from the perspective of Informativeness. Don't have a preference for long text. Score is from 1.0 to 5.0."}, 
    #         {"role": "user", "content": f"Dialog: {hyp_dialog_str}\nInformativeness Score:"}],
    #     temperature = 0.2,
    #     max_tokens = 20,
    #     ))

    while True:
        try:
            empathy_score = extract_num(get_response(
                [{"role": "system", "content": "You are an emotional expert, good at judging responses in emotional dialogues. Please score the bot's responses from the perspective of Empathy and Emotion Support. Don't have a preference for long text. Score is from 1.0 to 5.0. Always output the score first."}, 
                {"role": "user", "content": f"Dialog: {hyp_dialog_str}\nEmpathy Score (1.0~5.0): "}],
                temperature = 0.2,
                top_p = 0.5,
                max_tokens = 20,
                ))
            break
        except IndexError as e:
            print(e)
            continue

    return {
        # 'win_rate_utterance': win_rate_utterance,
        'match_ref_score': match_ref_score,
        'coherence_score': coherence_score,
        'informativeness_score': informativeness_score,
        # 'win_rate_dialog': win_rate_dialog,
        'empathy_score': empathy_score
    }

def worker(args):
    h, r = args
    return eval_dialog_by_gpt(h, r)

def eval_by_gpt_avg(hyp_dialogs, ref_dialogs):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        scores = pool.map(worker, zip(hyp_dialogs, ref_dialogs))

    avg_scores = {key: sum([score[key] for score in scores]) / len(scores) for key in scores[0]}
    return avg_scores


# TODO: Classification

# TODO: Extraction

def to_2d_list(one_d_list, sublist_sizes, add_speaker):
    two_d_list = []
    index = 0
    for size in sublist_sizes:
        if index + size > len(one_d_list):
            if add_speaker:
                two_d_list.append([('User: ' + x[0], 'Bot: ' + x[1][:min(x[1], 100)]) for x in one_d_list[index:]])
            else:
                two_d_list.append(one_d_list[index:])
            break
        if add_speaker:
            two_d_list.append([('User: '+ x[0], 'Bot: ' + x[1][:min(len(x[1]), 100)]) for x in one_d_list[index:index + size]])
        else:
            two_d_list.append(one_d_list[index:index + size])
        index += size
    return two_d_list


if __name__ == '__main__':
    s_time = time.time()

    # hypothesis = ["the cat was found under the bed"]
    # reference = ["the cat was under the bed"]
    # results = calculate_metrics_avg(hypothesis, reference)
    # for key, value in results.items():
    #     print(f"{key}: {value}")
    ref_dialog = [
        ["I there, dont know what to do, jst broke up with my girlfirned, we were 8 years together", "sorry to hear! do you have any idea about the break up? did you think about it ?"],
        ["Yes we decided together with our minds, and know i come home and feel so distant from the world", "sorry again! hope you'll get relief from this sadness. Please concentrate on your interests to divert your mind from this."],
        ["I couldn't wait to go to the concert.", "Which Concert?"],
        ["The U2 concert. Tickets were really expensive and I never thought we would be able to go, but somehow we did!!!", "Wow, that's awesome! I have never been to an actual concert."]
    ]
    hyp_dialog = [
        ["I there, dont know what to do, jst broke up with my girlfirned, we were 8 years together", "I am sorry to hear that. Do you have any idea about the break up?"],
        ["Yes we decided together with our minds, and know i come home and feel so distant from the world", "Hope you'll get relief from this sadness. Concentrating on your interests may divert your mind from this."],
        ["I couldn't wait to go to the concert.", "Wow~ Which Concert?"],
        ["The U2 concert. Tickets were really expensive and I never thought we would be able to go, but somehow we did!!!", "Wow, that's awesome! I have never been to an actual concert."	]
    ]

    import pickle

    with open("llama2_output_batch.pkl", "rb") as file:
        hyp_dialog = pickle.load(file)

    # ['user: ... bot', '...'] -> ['user: ...', 'bot: ...']
    ref_dialog = process_esconv(split='test')
    new_ref = []

    hyp_dialog = to_2d_list(hyp_dialog, [len(x) for x in ref_dialog], add_speaker=True)
    for dialog in ref_dialog:
        tmp = []
        for item in dialog:
            user, bot_agent = merged_string, last_element = '\n'.join(item[0].split('\n')[:-1]), item[0].split('\n')[-1]
            tmp.append([user, bot_agent + item[1]])
        new_ref.append(tmp)
    ref_dialog = new_ref
    # eval_dialog_by_gpt(hyp_dialog, ref_dialog)
    results = eval_by_gpt_avg(hyp_dialog[:100], ref_dialog[:100])
    print(results)
    
    print("Time cost: " + str(time.time() - s_time))