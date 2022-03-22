---
layout: post
categories: Competition
title:  "[Dacon] 한국어 문서 추출요약 AI 경진대회 참여기"
date:   2020-12-30
author: HaeYong Joung
tags: text-mining
comments: true
---

한국어 문서 추출요약 AI 경진대회
===============

2020년 겨울, Dacon에서 실시한 한국어 문서 추출요약 AI 대회에 참가했던 것을 기록해보고자 합니다.
(대회에 대한 자세한 사항은 [여기에!](https://dacon.io/competitions/official/235671/overview/))

*이러한 대회에 항상 함께해주는 쇠똥구리 팀원들에게 먼저 감사 인사를 전합니다.*

대회의 목적은 간단합니다! 다양한 한국어 기사 원문으로부터 적절한 **추출 요약문**을 도출해내는 모델을 만들면 됩니다.
먼저 데이터의 모습과 함께 모델링의 목적을 파악해보겠습니다.

- - -
### 데이터 & 분석 목적
![PNG](https://decision-J.github.io/assets/BertSum_Competition/data.PNG)

먼저 데이터는 약 20~30여개의 문장으로 이루어진 기사들입니다. 각 기사에는 **신문사, 기사원문, 요약문(label), 해당 요약문의 인덱스 값**이 포함되어 있습니다. train 데이터의 요약문의 경우 사람이 라벨링을 했다고 설명되어 있습니다.

인덱스가 3개인 것에서 눈치채셨겠지만 기사 원문에서 <u>가장 중요한 3개</u>의 문장을 추출하는 것이 대회의 목적입니다! 저희는 요약문을 추출하는 데 신문사 정보는 필요없다고 판단하여 원문과 요약문 데이터만을 가지고 모델링을 진행했습니다. 또한 학습 전에 특수 문자 제거, embedding 등 preprocessing 작업을 거쳤습니다. (embedding에는 뒤에서 언급할 koBert의 embedding 방법을 활용했습니다.)

- - -
### 사용 모델: koBert를 이용한 BertSum model
먼저 문서의 요약문을 만드는 방식에는 크게 두 가지가 있는 것 같습니다. 전체 텍스트에 적절한 요약문을 생성하는 **생성요약(Abstractive)**과 텍스트에 있는 문장 중 전체 내용을 대표한다고 생각되는 문장을 가져오는 **추출요약(Extractive)**이 있습니다. 저희 팀은 대회의 목적에 따라 추출요약에 해당하는 모델을 사용했습니다.

다양한 추출요약 모델 중에서 *Fine-tune BERT for Extractive Summarization(Yang Liu, 2019)* 논문을 참고했습니다. (편의상 BertSum 모델이라고 부르겠습니다.)
BertSum은 쉽게 말해서 구글의 pre-trained Bert 모델을 활용하여 텍스트 데이터를 학습하고 Transformer encoding을 통해 대표성을 갖는 문장을 classification해주는 방법입니다. 모델에 대한 더욱 상세하고 자세한 사항은 갓누누의 블로그, [이 곳](https://seonu-lim.github.io/nlp/BertSum/)을 참고해주시면 도움이 되실 겁니다!

![갓누누가 만든 모델 설명도](https://decision-J.github.io/assets/BertSum_Competition/bertsum.png "<u>갓누누</u>가 만든 모델 설명도")

앞 서 언급하였듯이 BertSum을 사용하려면 사전 학습된 Bert model이 필요한데요! 구글의 Bert는 영어를 기반으로 학습되어 있기 때문에 한국어를 기반한 모델이 필요했습니다. 다행히도 **SKT Brain**에서 한국어를 기반한 **koBert** 모형을 만들어주신 것을 알고 이 모형을 바탕으로 공모전을 진행했습니다.

- - -
### 모델링 특이사항
몇 개의 Baseline 모델을 만들어 본 후, 갓단단 팀원께서 한 가지 문제점을 발견했습니다. 몇몇 요약문들을 살펴보니 대표성이 없는 문장들이 추출된다는 것입니다. 예를 들어 다음과 같습니다.

- 단양 시청의 전경
- 삼성 라이온즈 사진 제공
- 신축 아파트 조감도
- ...

이러한 문장들이 대표 요약문으로 추출되는 이유를 다음 두가지 정도로 생각해보았습니다. 먼저, 기사와 관련된 강력한 키워드가 포함되어 있기 때문입니다. 예를 들어, "삼성 라이온즈 사진 제공"이라는 문장의 경우 기사 전반이 삼성 라이온즈 야구단에 관한 내용이기 때문에 해당 키워드가 직접적으로 포함되어 요약문으로 추출되었다는 것입니다.

두 번째로 데이터의 특성 때문입니다. 학습하고 있는 데이터가 기사이기 때문에 주로 두괄식으로 작성되어 있습니다. 따라서 1~3번째 문장에 핵심 요약문들이 포함된 경우가 많습니다. (위의 데이터 label index를 살펴보아도 0,1,2가 굉장히 많은 것을 알 수 있습니다.) 이 때, 몇몇 기사들은 맨 앞 문장에 기사에 포함된 사진에 관한 내용을 담는 경우가 있기 때문에 단순히 index에 따라서 저 문장들이 딸려 나온다는 것입니다.

이에 따라 갓단단 팀원은 <u>문장의 음절이 10개 이하인 문장</u>을 **사전에 제거**하고 모델링을 진행해봤다고 합니다. 그 결과 저런 터무니없는 요약문이 줄어들고 score 측면에서도 상당히 개선된 결과를 확인할 수 있었습니다.

- - -
### 결과
수 많은 파라미터 튜닝 과정들을 거쳐 최적의 모형을 구축하였고, 최종적으로 428팀 중 20위라는 결과를 기록했습니다. 개인적으로는 첫 NLP대회에서 만족할만한 결과를 거둬 팀원들에게 고맙고 뿌듯한 마음입니다!

- - -
### 참고 사이트

1. 갓누누 블로그: [https://seonu-lim.github.io/](https://seonu-lim.github.io/)
2. Fine-tune BERT for Extractive Summarization, Yang Liu, 2019: [https://arxiv.org/pdf/1903.10318.pdf](https://arxiv.org/pdf/1903.10318.pdf)
3. BertSum 저자 Github: [https://github.com/nlpyang/BertSum](https://github.com/nlpyang/BertSum)
4. SKT Brain koBert Github: [https://github.com/SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)
