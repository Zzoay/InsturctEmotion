

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics import accuracy_score


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

    # TODO: prompting
    num_turns = len(ref_dialog)     
    for hyp, ref in zip(hyp_dialog, ref_dialog):
        win_rate_utterance += 0.0 / num_turns
        match_ref_score += 0.0 / num_turns
        coherence_score += 0.0 / num_turns
        informativeness_score += 0.0 / informativeness_score

    win_rate_dialog = 0.0
    empathy_score = 0.0

    return {
        'win_rate_utterance': win_rate_utterance,
        'match_ref_score': match_ref_score,
        'coherence_score': coherence_score,
        'informativeness_score': informativeness_score,
        'win_rate_dialog': win_rate_dialog,
        'empathy_score': empathy_score
    }

def eval_by_gpt_avg(hyp_dialogs, ref_dialogs):
    scores = [eval_dialog_by_gpt(h, r) for h, r in zip(hyp_dialogs, ref_dialogs)]
    avg_scores = {key: sum([score[key] for score in scores]) / len(scores) for key in scores[0]}
    return avg_scores

# TODO: Classification

# TODO: Extraction


if __name__ == '__main':
    hypothesis = ["the cat was found under the bed"]
    reference = ["the cat was under the bed"]
    results = calculate_metrics_avg(hypothesis, reference)
    for key, value in results.items():
        print(f"{key}: {value}")