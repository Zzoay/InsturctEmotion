# EmoLA
Fine-tuning Llama2 on open-source emotion data.
We have collect several high-quality emotional data, including tasks of **emotion recognition, emotion support conversation, sentiment analysis, persona extraction, and commonsense learning**.
We unify these tasks into instruction tuning.
## Fine-tuning
```
deepspeed \
  --include localhost:{GPU_ID} train.py \
  --lora_hyperparams_file config/lora.json \
  --deepspeed config/deepspeed.json
```
## Todo List
- [x] Basic fine-tuning (PEFT) and inference.
- [ ] Full fine-tuning.
- [ ] Auto Evaluation (GPT-3.5, GPT-4, Claude).
- [ ] More data.

## Data List
### Emotion Recognition
GoEmotions
```bibtex
@inproceedings{demszky-2020-goemotions,
 author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
 booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
 title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
 year = {2020}
}
```
EmoWoZ
```bibtex
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
```
AffectiveText
```bibtex
@inproceedings{strapparava-mihalcea-2007-semeval,
    title = "SemEval-2007 Task 14: Affective Text",
    author = "Strapparava, Carlo  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007)",
    month = jun,
    year = "2007",
    address = "Prague, Czech Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S07-1013",
    pages = "70--74",
}
```
WASSA2017
```bibtex
@inproceedings{mohammad-bravo-marquez-2017-wassa,
    title = "WASSA-2017 Shared Task on Emotion Intensity",
    author = "Mohammad, Saif  and
      Bravo-Marquez, Felipe",
    booktitle = "Proceedings of the 8th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5205",
    doi = "10.18653/v1/W17-5205",
    pages = "34--49",
}
```
SemEval2018
```bibtex
@InProceedings{SemEval2018Task1,
 author = {Mohammad, Saif M. and Bravo-Marquez, Felipe and Salameh, Mohammad and Kiritchenko, Svetlana},
 title = {SemEval-2018 {T}ask 1: {A}ffect in Tweets},
 booktitle = {Proceedings of International Workshop on Semantic Evaluation (SemEval-2018)},
 address = {New Orleans, LA, USA},
 year = {2018}}
```
MELD
```bibtex
@inproceedings{poria-etal-2019-meld,
    title = "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations",
    author = "Poria, Soujanya  and
      Hazarika, Devamanyu  and
      Majumder, Navonil  and
      Naik, Gautam  and
      Cambria, Erik  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1050",
    doi = "10.18653/v1/P19-1050",
    pages = "527--536",
}
```
EmoContext
```bibtex
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
}
```
EmotionTweetEval
```bibtex
@inproceedings{barbieri2020tweeteval,
  title={{TweetEval:Unified Benchmark and Comparative Evaluation for Tweet Classification}},
  author={Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
  booktitle={Proceedings of Findings of EMNLP},
  year={2020}
}
```
IEMOCAP
```bibtex
@inproceedings{zadeh-lpvcm-2018,
  author       = {Amir Zadeh and
                  Paul Pu Liang and
                  Soujanya Poria and
                  Prateek Vij and
                  Erik Cambria and
                  Louis{-}Philippe Morency},
  editor       = {Sheila A. McIlraith and
                  Kilian Q. Weinberger},
  title        = {Multi-attention Recurrent Network for Human Communication Comprehension},
  booktitle    = {Proceedings of the Thirty-Second {AAAI} Conference on Artificial Intelligence,
                  (AAAI-18), the 30th innovative Applications of Artificial Intelligence
                  (IAAI-18), and the 8th {AAAI} Symposium on Educational Advances in
                  Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February
                  2-7, 2018},
  pages        = {5642--5649},
  publisher    = {{AAAI} Press},
  year         = {2018},
  url          = {https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17390},
  timestamp    = {Tue, 08 Mar 2022 21:46:35 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZadehLPVCM18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

### Emotion Support (Empathetic) Conversation
[ESConv](https://huggingface.co/datasets/thu-coai/esconv)
```bibtex
@inproceedings{LiuZDSLYJH20,
  author       = {Siyang Liu and
                  Chujie Zheng and
                  Orianna Demasi and
                  Sahand Sabour and
                  Yu Li and
                  Zhou Yu and
                  Yong Jiang and
                  Minlie Huang},
  editor       = {Chengqing Zong and
                  Fei Xia and
                  Wenjie Li and
                  Roberto Navigli},
  title        = {Towards Emotional Support Dialog Systems},
  booktitle    = {Proceedings of the 59th Annual Meeting of the Association for Computational
                  Linguistics and the 11th International Joint Conference on Natural
                  Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
                  Event, August 1-6, 2021},
  pages        = {3469--3483},
  publisher    = {Association for Computational Linguistics},
  year         = {2021},
  url          = {https://doi.org/10.18653/v1/2021.acl-long.269},
  doi          = {10.18653/v1/2021.acl-long.269},
  timestamp    = {Tue, 27 Jun 2023 15:48:45 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LiuZDSLYJH20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
[EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)
```bibtex
@inproceedings{rashkin-etal-2019-towards,
    title = "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset",
    author = "Rashkin, Hannah  and
      Smith, Eric Michael  and
      Li, Margaret  and
      Boureau, Y-Lan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1534",
    doi = "10.18653/v1/P19-1534",
    pages = "5370--5381",
}

```
[AugESC](https://huggingface.co/datasets/thu-coai/augesc)
```bibtex
@inproceedings{ZhengSW0H23,
  author       = {Chujie Zheng and
                  Sahand Sabour and
                  Jiaxin Wen and
                  Zheng Zhang and
                  Minlie Huang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {AugESC: Dialogue Augmentation with Large Language Models for Emotional
                  Support Conversation},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {1552--1568},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.99},
  doi          = {10.18653/v1/2023.findings-acl.99},
  timestamp    = {Thu, 10 Aug 2023 12:35:52 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ZhengSW0H23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

### Sentiment Analysis
[ASPE](https://github.com/NUSTM/ChatGPT-Sentiment-Evaluation/tree/main/data/open_domain/ASPE)
```bibtex
@article{wang2023chatgptsentiment,
  title={Is ChatGPT a Good Sentiment Analyzer? A Preliminary Study},
  author={Zengzhi Wang and Qiming Xie and Zixiang Ding and Yi Feng and Rui Xia},
  journal={arXiv preprint},
  year={2023}
}
```

[DMASTE](https://github.com/NJUNLP/DMASTE)
```bibtex
@inproceedings{XuYWCZD23,
  author       = {Ting Xu and
                  Huiyun Yang and
                  Zhen Wu and
                  Jiaze Chen and
                  Fei Zhao and
                  Xinyu Dai},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Measuring Your {ASTE} Models in The Wild: {A} Diversified Multi-domain
                  Dataset For Aspect Sentiment Triplet Extraction},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {2837--2853},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.178},
  doi          = {10.18653/v1/2023.findings-acl.178},
  timestamp    = {Thu, 10 Aug 2023 12:35:54 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/XuYWCZD23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

[ACOS](https://github.com/NUSTM/ACOS)
```bibtex
@inproceedings{cai2021aspect,
  title={Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions},
  author={Cai, Hongjie and Xia, Rui and Yu, Jianfei},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={340--350},
  year={2021}
}

```
[MEMD-ABSA](https://github.com/NUSTM/MEMD-ABSA)
```bibtex
@article{cai2023memd,
  title={MEMD-ABSA: A Multi-Element Multi-Domain Dataset for Aspect-Based Sentiment Analysis},
  author={Cai, Hongjie and Song, Nan and Wang, Zengzhi and Xie, Qiming and Zhao, Qiankun and Li, Ke and Wu, Siwei and Liu, Shijie and Yu, Jianfei and Xia, Rui},
  journal={arXiv preprint arXiv:2306.16956},
  year={2023}
}
```
### Persona Extraction and Commonsense Knowledge Learning
[PAED](https://github.com/Cyn7hia/PAED/tree/main)
```bibtex
@inproceedings{zhu2023paed,
  title = {PAED: Zero-Shot Persona Attribute Extraction in Dialogues},
  author = {Zhu, Luyao and Li, Wei and Mao, Rui and Pandelea, Vlad and Cambria, Erik},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year = {2023}
}
```
[PeaCoK](https://github.com/Silin159/PeaCoK)
```bibtex
@inproceedings{GaoBOBKWMB23,
  author       = {Silin Gao and
                  Beatriz Borges and
                  Soyoung Oh and
                  Deniz Bayazit and
                  Saya Kanno and
                  Hiromi Wakaki and
                  Yuki Mitsufuji and
                  Antoine Bosselut},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {PeaCoK: Persona Commonsense Knowledge for Consistent and Engaging
                  Narratives},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                  July 9-14, 2023},
  pages        = {6569--6591},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.acl-long.362},
  doi          = {10.18653/v1/2023.acl-long.362},
  timestamp    = {Thu, 10 Aug 2023 12:35:47 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/GaoBOBKWMB23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
[Atomic](https://huggingface.co/datasets/atomic)
```bibtex
@article{Sap2019ATOMICAA, 
title={ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning}, 
author={Maarten Sap and Ronan Le Bras and Emily Allaway and Chandra Bhagavatula and Nicholas Lourie and Hannah Rashkin and Brendan Roof and Noah A. Smith and Yejin Choi}, 
journal={ArXiv}, 
year={2019}, 
volume={abs/1811.00146} 
}
```
## Citation
If you feel it is useful, please cite the data you used and this item:
```bibtex
@misc{jiang-emola-2023,
    author = {Gongyao Jiang},
    title = {EmoLA: Fine-tuning Llama2 on open-source emotion data},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/Zzoay/EmoLA}}
}
```