
import re
import json
import random

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import codecs as cs
import pandas as pd


class DatasetProcessorGoEmotions:
    _DESCRIPTION = """\
        The GoEmotions dataset contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.
        The emotion categories are admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
        disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness,
        optimism, pride, realization, relief, remorse, sadness, surprise.
        """

    _CITATION = """\
        @inproceedings{demszky2020goemotions,
         author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
         booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
         title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
         year = {2020}
        }
        """

    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        # load from cache or remote
        # dataset = load_dataset(path='go_emotions', name='simplified', split=self.data_type)
        # --- load from disk ---
        # if load_from_disk, will cause KeyError of 'length'. Unknown reason.
        dataset = load_from_disk(f"data/emotion_recognition/GoEmotions/{self.data_type}")
        # https://github.com/google-research/google-research/tree/master/goemotions/data
        # dataset = pd.read_csv(f"data/emotion_recognition/go_emotions/{self.data_type}.tsv", delimiter="\t", names=['text', 'labels', 'id'])
        # dataset['labels'] = dataset['labels'].apply(lambda x: list(map(int, x.split(','))))
        # --------------
        _CLASS_NAMES = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        for i in tqdm(range(len(dataset))):
            results['input_text'].append(dataset[i]['text'])
            label = dataset[i]['labels']
            new_label = []
            for gold_id in label:
                new_label.append(_CLASS_NAMES[gold_id])
            results['output_text'].append(new_label)
        return results


class DatasetProcessorAffectiveText: # SemEval-2007 (Affective Text) ; No validation
    def __init__(self, data_type):

        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        # dataset = load_dataset(path='israelfama/semeval2007_task_14', split=self.data_type)
        dataset = load_from_disk('data/emotion_recognition/AffectiveText')[self.data_type]
        _CLASS_NAMES = [
            "None",
            "neutral",
            "fearful",
            "dissatisfied",
            "apologetic",
            "abusive",
            "excited",
            "satisfied",
        ]

        for i in tqdm(range(len(dataset))):
            results['input_text'].append(dataset['text'][i])
            results['output_text'].append(dataset['label'][i])
        return results


class DatasetProcessorWASSA2017: # WASSA-2017/EmoInt ; No validation
    def __init__(self, data_type):

        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': [], 'intensity': []}
        # dataset = load_dataset(path='stepp1/tweet_emotion_intensity', split=self.data_type)
        dataset = load_from_disk('data/emotion_recognition/WASSA2017')[self.data_type]
        _CLASS_NAMES = [
            "None",
            "neutral",
            "fearful",
            "dissatisfied",
            "apologetic",
            "abusive",
            "excited",
            "satisfied",
        ]

        for i in tqdm(range(len(dataset))):
            text = re.sub("[@#]\S+", '', dataset[i]['tweet'])
            text = text.replace("&amp;", "").replace('\\n', ' ').strip()
            results['input_text'].append(text)
            label = random.choice([
                f"{dataset[i]['sentiment_intensity']}-level {dataset[i]['class']}", 
                f"{dataset[i]['class']} with {dataset[i]['sentiment_intensity']} level"
            ])
            results['output_text'].append(label)
            results['intensity'].append(dataset[i]['sentiment_intensity'])
        return results


class DatasetProcessorSemEval2018:
    _CITATION = """\
    @InProceedings{SemEval2018Task1,
     author = {Mohammad, Saif M. and Bravo-Marquez, Felipe and Salameh, Mohammad and Kiritchenko, Svetlana},
     title = {SemEval-2018 {T}ask 1: {A}ffect in Tweets},
     booktitle = {Proceedings of International Workshop on Semantic Evaluation (SemEval-2018)},
     address = {New Orleans, LA, USA},
     year = {2018}}
    """

    _DESCRIPTION = """\
     SemEval-2018 Task 1: Affect in Tweets: SubTask 5: Emotion Classification.
     This is a dataset for multilabel emotion classification for tweets.
     'Given a tweet, classify it as 'neutral or no emotion' or as one, or more, of eleven given emotions that best represent the mental state of the tweeter.'
     It contains 22467 tweets in three languages manually annotated by crowdworkers using Best–Worst Scaling.
    """

    _HOMEPAGE = "https://competitions.codalab.org/competitions/17751"

    _LICENSE = ""

    _URLs = {
        "subtask5.english": ["https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"],
        "subtask5.spanish": ["https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"],
        "subtask5.arabic": ["https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"],
    }

    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        # dataset = load_dataset(path='sem_eval_2018_task_1', split=self.data_type, name='subtask5.english')
        dataset = load_from_disk(f'data/emotion_recognition/SemEval2018/{self.data_type}')
        _CLASS_NAMES = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "love",
            "optimism",
            "pessimism",
            "sadness",
            "surprise",
            "trust"
        ]

        for i in tqdm(range(len(dataset))):
            text = re.sub("[@#]\S+", '', dataset[i]['Tweet'])
            text = text.replace("&amp;", "").replace("\\n", " ").strip()
            
            new_label = []
            for emotion in _CLASS_NAMES:
                if dataset[i][emotion]:
                    new_label.append(emotion)
            
            if len(new_label) > 0:
                results['input_text'].append(text)
                results['output_text'].append(new_label)
        return results


class DatasetProcessorMELD:
    def __init__(self, data_type):
        if data_type == 'train':
            data_type = 'train_sent_emo.csv'
        elif data_type == 'test':
            data_type = 'test_sent_emo.csv'
        elif data_type == 'validation':
            data_type = 'dev_sent_emo.csv'
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        data = pd.read_csv(f"data/emotion_recognition/MELD/{self.data_type}")
        pre_dial_id = -1
        dialogue = []
        label = []
        dialogs, labels = [], []
        for row in tqdm(data.iterrows()):
            meta = row[1]
            text = meta['Utterance'].replace('’', '\'').replace("\"", '')
            speaker = meta['Speaker']
            emotion = meta['Emotion'].lower()
            turn_data = {}
            turn_data['speaker'] = speaker
            turn_data['text'] = text
            # turn_data['label'] = emotion

            dialogue_id = meta['Dialogue_ID']
            if pre_dial_id == -1:
                pre_dial_id = dialogue_id
            if dialogue_id != pre_dial_id:
                # results['input_text'].append(dialogue)
                # results['output_text'].append(label)
                dialogs.append(dialogue)
                labels.append(label)
                dialogue = []
                label = []
            pre_dial_id = dialogue_id
            dialogue.append(': '.join(turn_data.values()))
            label.append(emotion)
        # results['input_text'].append(dialogue)
        # results['output_text'].append(label)
        dialogs.append(dialogue)
        labels.append(label)
        for i in range(len(dialogs)):
            dialog, label = dialogs[i], labels[i]
            for j in range(len(dialog)):  
                if label[j] == 'neutral' and random.random() < 0.8:
                    continue
                offset = 0
                if j > 5:  # if too long, keep 6 turns
                    offset = j - 5
                results['input_text'].append(dialog[offset:j + 1])
                results['output_text'].append(label[j])
        return results


class DatasetProcessorEmoContext: # SemEval-2019 Task 3
    _CITATION = """\
    @inproceedings{chatterjee-etal-2019-semeval,
        title={SemEval-2019 Task 3: EmoContext Contextual Emotion Detection in Text},
        author={Ankush Chatterjee and Kedhar Nath Narahari and Meghana Joshi and Puneet Agrawal},
        booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation},
        year={2019},
        address={Minneapolis, Minnesota, USA},
        publisher={Association for Computational Linguistics},
        url={https://www.aclweb.org/anthology/S19-2005},
        doi={10.18653/v1/S19-2005},
        pages={39--48},
        abstract={In this paper, we present the SemEval-2019 Task 3 - EmoContext: Contextual Emotion Detection in Text. Lack of facial expressions and voice modulations make detecting emotions in text a challenging problem. For instance, as humans, on reading ''Why don't you ever text me!'' we can either interpret it as a sad or angry emotion and the same ambiguity exists for machines. However, the context of dialogue can prove helpful in detection of the emotion. In this task, given a textual dialogue i.e. an utterance along with two previous turns of context, the goal was to infer the underlying emotion of the utterance by choosing from four emotion classes - Happy, Sad, Angry and Others. To facilitate the participation in this task, textual dialogues from user interaction with a conversational agent were taken and annotated for emotion classes after several data processing steps. A training data set of 30160 dialogues, and two evaluation data sets, Test1 and Test2, containing 2755 and 5509 dialogues respectively were released to the participants. A total of 311 teams made submissions to this task. The final leader-board was evaluated on Test2 data set, and the highest ranked submission achieved 79.59 micro-averaged F1 score. Our analysis of systems submitted to the task indicate that Bi-directional LSTM was the most common choice of neural architecture used, and most of the systems had the best performance for the Sad emotion class, and the worst for the Happy emotion class}
    }
    """

    _DESCRIPTION = """\
    In this dataset, given a textual dialogue i.e. an utterance along with two previous turns of context, the goal was to infer the underlying emotion of the utterance by choosing from four emotion classes - Happy, Sad, Angry and Others.
    """
    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        # dataset = load_dataset(path='emo', split=self.data_type)
        dataset = load_from_disk(f'data/emotion_recognition/EmoContext/{self.data_type}')
        _CLASS_NAMES = [
            "others",
            "happy",
            "sad",
            "angry"
        ]

        for i in tqdm(range(len(dataset))):
            label = dataset[i]['label']
            new_label = _CLASS_NAMES[label]
            if new_label == 'others' and random.random() < 0.8:
                continue
            results['input_text'].append("Context: " + dataset[i]['text'])
            results['output_text'].append(new_label)
        return results


class DatasetProcessorEmoryNLP:
    def __init__(self, data_type):
        if data_type == 'train':
            data_type = 'emotion-detection-trn.json'
        elif data_type == 'test':
            data_type = 'emotion-detection-tst.json'
        elif data_type == 'validation':
            data_type = 'emotion-detection-dev.json'
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        data = json.load(open(f"data/emotion_recognition/EmoryNLP/{self.data_type}", 'r', encoding='utf8'))
        _CLASS_NAMES = {
            "hap": "happiness",
            "ang": "anger",
            "sad": "sadness",
            "neu": "neutral",
        }

        for episode in tqdm(data['episodes']):
            for scene in episode['scenes']:
                dialogue = []
                label = []
                for i, utterance in enumerate(scene['utterances']):
                    text = utterance['transcript']
                    speaker = utterance['speakers'][0]
                    speaker = speaker.split(' ')[0]
                    emotion = utterance['emotion'].lower()
                    turn_data = {}
                    turn_data['speaker'] = speaker
                    turn_data['text'] = text
                    label.append(emotion)
                    dialogue.append(': '.join(turn_data.values()))

                    # drop 50% data with neutral labels
                    if emotion == 'neutral' and random.random() < 0.5:
                        continue

                    offset = 0
                    if i > 7:  # keep 8 turns
                        offset = i - 7
                    results['input_text'].append(dialogue[offset:i+1])
                    results['output_text'].append(emotion)
        return results


class DatasetProcessorEmotionTweetEval:
    _CITATION = """\
    @inproceedings{barbieri2020tweeteval,
      title={{TweetEval:Unified Benchmark and Comparative Evaluation for Tweet Classification}},
      author={Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
      booktitle={Proceedings of Findings of EMNLP},
      year={2020}
    }
    """

    _DESCRIPTION = """\
    TweetEval consists of seven heterogenous tasks in Twitter, all framed as multi-class tweet classification. All tasks have been unified into the same benchmark, with each dataset presented in the same format and with fixed training, validation and test splits.
    """

    _HOMEPAGE = "https://github.com/cardiffnlp/tweeteval"

    _LICENSE = ""

    URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/"
    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        # dataset = load_dataset(path='tweet_eval', split=self.data_type, name='emotion')
        dataset = load_from_disk(f'data/emotion_recognition/EmotionTweetEval/{self.data_type}')
        _CLASS_NAMES = [
            "anger",
            "joy",
            "optimism",
            "sadness"
        ]

        for i in range(len(dataset)):
            results['input_text'].append(dataset[i]['text'])
            label = dataset[i]['label']
            new_label = [_CLASS_NAMES[label]]
            results['output_text'].append(new_label)
        return results


class DatasetProcessorSentimentTweetEval:
    _CITATION = """\
    @inproceedings{barbieri2020tweeteval,
      title={{TweetEval:Unified Benchmark and Comparative Evaluation for Tweet Classification}},
      author={Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
      booktitle={Proceedings of Findings of EMNLP},
      year={2020}
    }
    """

    _DESCRIPTION = """\
    TweetEval consists of seven heterogenous tasks in Twitter, all framed as multi-class tweet classification. All tasks have been unified into the same benchmark, with each dataset presented in the same format and with fixed training, validation and test splits.
    """

    _HOMEPAGE = "https://github.com/cardiffnlp/tweeteval"

    _LICENSE = ""

    URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/"
    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        dataset = load_dataset(path='tweet_eval', split=self.data_type, name='sentiment')
        _CLASS_NAMES = [
            "negative",
            "neutral",
            "positive",
        ]

        for i in range(len(dataset)):
            results['input_text'].append(dataset['text'][i])
            label = dataset['label'][i]
            new_label = [_CLASS_NAMES[label]]
            results['output_text'].append(new_label)
        return results


class DatasetProcessorSILICONE:
    pass


class DatasetProcessorIEMOCAP:
    def __init__(self, data_type):
        if data_type == 'train':
            data_type = 'train_data.json'
        elif data_type == 'test':
            data_type = 'test_data.json'
        elif data_type == 'validation':
            data_type = 'dev_data.json'
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results['input_text'][index]
        output_text = self.results['output_text'][index]

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        results = {'input_text': [], 'output_text': []}
        data = json.load(open(f"data/emotion_recognition/IEMOCAP/{self.data_type}", 'r', encoding='utf8'))
        _CLASS_NAMES = {
            "hap": "happiness",
            "ang": "anger",
            "sad": "sadness",
            "neu": "neutral",
            "fru": "frustration",
            "exc": 'excitement'
        }
        f = cs.open('data/emotion_recognition/IEMOCAP/name_map', 'r', encoding='utf-8').readlines()
        name_maps = eval(f[0])
        for dialog in tqdm(data):
            dialogue = []
            label = []
            for i, utterance in enumerate(dialog):
                speaker = utterance.get('speaker')
                text = utterance.get('text').replace('[LAUGHTER]', '')
                emotion = utterance.get('label')
                if emotion is not None:
                    emotion = _CLASS_NAMES[emotion]
                else:
                    emotion = 'none'
                speaker = name_maps[speaker]
                turn_data = {}
                turn_data['speaker'] = speaker
                turn_data['text'] = text
                label.append(emotion)
                dialogue.append(": ".join(turn_data.values()))

                # drop 50 % data with neutral labels
                if emotion == 'none' or (emotion == 'neutral' and random.random() < 0.5):
                    continue

                offset = 0
                if i > 9:  # keep 10 turns
                    offset = i - 9
                results['input_text'].append(dialogue[offset:i+1])
                results['output_text'].append(emotion)
        return results


class DatasetProcessorEmoWoz:
    _CITATION = """\
    @inproceedings{feng-etal-2022-emowoz,
        title = "{E}mo{WOZ}: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems",
        author = "Feng, Shutong  and
          Lubis, Nurul  and
          Geishauser, Christian  and
          Lin, Hsien-chin  and
          Heck, Michael  and
          van Niekerk, Carel  and
          Gasic, Milica",
        booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
        month = jun,
        year = "2022",
        address = "Marseille, France",
        publisher = "European Language Resources Association",
        url = "https://aclanthology.org/2022.lrec-1.436",
        pages = "4096--4113",
    }
    """

    _DESCRIPTION = """\
    EmoWOZ is a user emotion recognition in task-oriented dialogues dataset, \
    consisting all dialogues from MultiWOZ and 1000 additional human-machine \
    dialogues (DialMAGE). Each user utterance is annotated with one of the \
    following emotions: 0: neutral, 1: fearful, 2: dissatisfied, 3: apologetic, \
    4: abusive, 5: excited, 6: satisfied. System utterances are annotated with \
    -1. For detailed label design and explanation, please refer to the paper and \
    dataset homepage.
    """

    _HOMEPAGE = "https://zenodo.org/record/6506504"

    _LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/legalcode"

    _URLS = {
        "emowoz_multiwoz": "https://zenodo.org/record/6506504/files/emowoz-multiwoz.json",
        "emowoz_dialmage": "https://zenodo.org/record/6506504/files/emowoz-dialmage.json",
        "emowoz_split": "https://zenodo.org/record/6506504/files/data-split.json"
    }
    def __init__(self, data_type):
        self.data_type = data_type
        self.results = self.process_dataset()

    def __len__(self):
        return len(self.results['input_text'])

    def __getitem__(self, index):
        input_text = self.results[index]['input_text']
        output_text = self.results[index]['output_text']

        return {
            "input_text": input_text,
            "output_text": output_text
        }

    def process_dataset(self):
        # results = {'input_text': [], 'output_text': []}
        ret = []
        # dataset = load_dataset(path='hhu-dsml/emowoz', split=self.data_type, name='emowoz')
        dataset = load_from_disk(f'data/emotion_recognition/EmoWOZ/{self.data_type}')
        _CLASS_NAMES = [
            "neutral",
            "fearful",
            "dissatisfied",
            "apologetic",
            "abusive",
            "excited",
            "satisfied",
        ]

        for i in tqdm(range(len(dataset))):
            log = dataset[i]['log']
            text = log['text']
            uttrs = []
            for j, uttr in enumerate(text):
                if j % 2 == 0:
                    uttr = 'Participant 1: ' + uttr
                else:
                    uttr = 'Participant 2: ' + uttr
                uttrs.append(uttr)
        
            emotion = log['emotion']
            new_label = []
            for j, ids in enumerate(emotion):
                ids = int(ids)
                if ids == -1:
                    new_label.append('none')
                else:
                    # new_label.append(_CLASS_NAMES[ids])
                    label = _CLASS_NAMES[ids]
                    # drop 90% of the data with a neutral label
                    if label == 'neutral' and random.random() < 0.9:
                        continue
                    # results['input_text'].append(uttrs[:j+1])
                    # results['output_text'].append(label)
                    ret.append({'input_text': uttrs[:j+1], "output_text": label})
        # SA doesn't quite match emotion supporting, so drop 50% data.
        ret = [x for x in ret if random.random() < 0.5]
        return ret


# def process_emotionlines():
