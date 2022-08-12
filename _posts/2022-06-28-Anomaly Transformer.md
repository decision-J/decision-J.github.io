---
layout: post
categories: Papers
title:  "[Time Series] Anomaly Transformer"
date:   2022-06-28
author: HaeYong Joung
tags: TimeSeries Anomaly-detection Transformer Attention
comments: true
---

[Paper review] Anomaly Transformer
===============

시계열 데이터의 이상치 탐색 방법론으로서 Transformer 매커니즘을 활용한 **Anomaly Transformer**에 대해 리뷰해보겠습니다. ICLR 2022 의 spotlight 논문이라고 하네요!

*Xu, Jiehui, et al. "Anomaly transformer: Time series anomaly detection with association discrepancy." arXiv preprint arXiv:2110.02642 (2021)*

(본 리뷰의 모든 수식과 그림은 원 논문을 참고했습니다.)

- - -

### Idea
**Anomaly Transformer**는 이름에서 알 수 있듯이 *Transformer*를 활용하여 시계열 데이터에서의 *Anomaly detection*을 잘 해보자는 목적을 가지고 있습니다. 이상점을 탐색하는 방법으로는 reconstruction based 방법을 사용하고 있습니다. Unsupervised learning으로 주어진 train 시계열 데이터의 패턴을 통해 정상 시계열 패턴을 잘 재구축 하도록 모형을 학습시킵니다. 추후에 test set이 들어왔을 때, 모형이 재구축(reconstruction)한 시계열 패턴과의 비교를 통해 그 차이가 큰 (anomaly score가 큰) 지점을 이상점으로 탐색하는 방법론입니다. 이 때, 정상 시계열 패턴을 잘 학습하도록 해주기 위해 Transformer를 활용합니다. Self-attention을 이상점 탐지에 특화되게 바꾼 Anomaly-attention을 제안하였습니다. 그럼 Anomaly Transformer에 대해 본격적으로 알아보도록 하겠습니다. 

### Anomaly Transformer
Anomaly Transformer의 architecture는 다음과 같습니다.

<p align="center">
  <img src="https://decision-J.github.io/assets/Anomaly_Detection/Anomaly_Transformer/Architecture.png" alt="Anomaly Transformer architecture"/>
</p>

Input embedding $X_{0}$가 들어오면 크게 **Anomaly Attenion** layer와 Feed Forward layer를 L번 반복 수행하며 input time series의 pattern을 학습하게 됩니다. 이후 이 학습된 정보를 가지고 가장 보편적인 pattern의 reconstruction data를 산출하는데요! 이 구조에서 가장 주목해야 하고, Anomaly transformer가 높은 성능을 가지는 이유는 단연 Anomaly attention일 것입니다. 

### Anomaly Attention
Anomaly attention은 기본적인 attention을 가지고 저자들이 time series anomaly detection에 맞게 매커니즘을 조금 수정한 형태입니다. 가장 큰 부분은 attention 내부를 *Prior-Association*과 *Series-Association*의 두 가지로 나누었다는 것입니다. 
먼저 prior association은 이름에서 알 수 있듯이, 모형에서 사용하는 Gaussian kernel의 시그마를 학습합니다. Gaussian Kernel은 attention의 Query, Key를 학습할 때 영향을 줍니다. 커널을 통해 다양한 패턴의 시계열 자료에 적용할 수 있다고 저자는 설명하고 있습니다. Series asssociation은 Query와 Key를 학습하는 부분입니다. 저자들은 이 두 association을 통해서(정확히는 prior association) 시계열 자료를 point-wise가 아닌, temporal dependency를 학습할 수 있다고 설명합니다. Gaussian Kernel의 $\sigma$를 통해서 인접한 time point에 더 큰 가중치를 줄 수 있다는 것이지요.

### Association Discrepancy
Association Discrepancy는 앞서 살펴본 prior association과 series association간의 차이를 계산하는 것입니다. 차이를 계산하는 방법은 KL divergence를 활용하였으며, KL divergence의 assymetric한 점을 보완하기 위해 순서를 바꿔가며 계산한 평균을 사용하였습니다. Anomaly attention은 이 지표를 활용하여 가중치를 업데이트 하게 됩니다.

$$ AssDis(P,S; X) = \frac{1}{L} \sum^L_{l=1}(KL(P^l_i||S^l_i) + KL(S^l_i||P^l_i)) $$
$$ where\,\, i=1,...,N $$


### Mini Max Association Learning
- Mini Max Strategy
- Association-based Anomaly Criterion
  
  
### Experiments

### 느낀점





- - -
### Reference

1. Paper: [Anomaly Transformer](https://arxiv.org/pdf/2110.02642.pdf)
2. Youtube: [고려대학교 산업경영공학과 DSBA연구실 세미나](https://www.youtube.com/watch?v=C3dphckvyn0&ab_channel=%EA%B3%A0%EB%A0%A4%EB%8C%80%ED%95%99%EA%B5%90%EC%82%B0%EC%97%85%EA%B2%BD%EC%98%81%EA%B3%B5%ED%95%99%EB%B6%80DSBA%EC%97%B0%EA%B5%AC%EC%8B%A4)
3. Github: [Code (Pytorch)](https://github.com/thuml/Anomaly-Transformer) 








### Architeture
Efficient Det은 구글 브레인의 이전 논문인 Efficient Net을 Backbone으로 활용한 Object detection 버전 모델입니다. 논문 곳곳에서 효율적이고 가볍다는 언급을 자주 할만큼 효율성에 집중한 듯 하지만 정확도 또한 20년 기준 SOTA를 달성한 굉장한 모형입니다. 기본적으로 YOLO 등과 비슷하게 One stage detection 구조를 가지고 있습니다. 자세한 아키텍처는 아래와 같이 표현할 수 있습니다.

$$ Efficient\, Det = BiFPN + Compound\, Scaling + Efficient\, Net$$

<p align="center">
  <img src="https://decision-J.github.io/assets/computer_vision/Efficient_Det/architecture.png" alt="Efficient Det architecture"/>
</p>

Backbone인 Effcient Net외에 *Bi-FPN*과 *Compound Scaling*이라는 두 기법이 추가로 더 적용되어 있는 것을 확인하실 수 있습니다. 지금부터 두 방법이 어떤 것인지 알아보도록 하겠습니다.

### Bi-FPN
<p align="center">
  <img src="https://decision-J.github.io/assets/computer_vision/Efficient_Det/bifpn.png" alt="Bi-FPN과 기존 Feature pyramid 방식들"/>
</p>

기본적으로 Object detection 알고리즘들은 이미지나 비디오의 *feature*를 기준으로 detecting을 합니다. 이 때, 다양한 resolution에서 feature를 받아 분석하는 것을 **multi-scale feature fusion** 이라고 합니다. 대표적인 방식으로는 FPN(Feature Pyramid Network)이 있죠! FPN은 각 scale에서 feature를 받을 수 있지만 Top-Down 방식만 가능하다는 단점이 있습니다. 이에 Bottom-Up flow도 추가한 것이 PANet입니다. 가장 정확하다는 평가를 받는 feature fusion 방식입니다. 

하지만 이러한 PANet은 굉장히 많은 parameter를 가질 수 밖에 없기 때문에 cost가 늘어난다는 단점이 있습니다. (이 단점을 해결하기 위해 NAS-FPN이 등장했지만 정확도가 떨어집니다.) Bi-FPN은 본 논문에서 제시하는 방법론으로, 정확성을 담보하면서도 PANet보다 효율적인 구조입니다. 개인적으로는 PANet 기본 구조에 효율성을 갖도록 몇 가지를 수정한 방법이라고 보여집니다. 

우선 Bi-FPN의 Scale connection 부분부터 살펴보겠습니다. Bi-FPN은 PANet과는 달리 input edge만을 갖는 노드는 없게 설계하여 parameter를 줄였습니다. 또한 Same level의 input node는 바로 bottom-up 단계의 output node로 연결하여 많은 cost없이 feature fusion이 일어나도록 설계했습니다. 마지막으로 Top-down / Bottom-up path들을 NAS-FPN처럼 반복함으로써 정확도를 확보했습니다.

이렇게 연결된 feature들을 fusion하는 방법에 대해 알아보겠습니다. 기본적으로 각 Input들을 그냥 결합할 경우 scale에 따라 가중치가 달라지기 때문에, 이를 반영하여 결합하는 것이 각 scale의 주요 특징들을 반영하는 데 중요합니다. 본 논문에서는 *Fast Normalized fusion*을 사용해서 빠르게 결합하는 방식을 사용했습니다.

$$ O = \sum_i \frac{w_i}{\epsilon+\sum_j w_j} * I_i$$

### Compound Scaling
앞 서 Bi-FPN에서 반복을 통해 feature들을 파악한다고 했는데요! 그러면 layer의 반복 수는 어떻게 정하면 될까요? 또 Efficient Det의 scale을 키울 때, 각각의 channel이나 layer들은 어떻게 정해야 효율적으로 scale up 할 수 있을까요? 이런 것들을 정하는 것이 **Compound Scaling** 파트입니다.

결론적으로 말하면 케이스 별로 optimize해주는 것은 아니고 저자들이 실험을 통해 최적의 궁합을 찾았다는 것입니다. 저자들은 *compound coefficient* $\phi$를 사용하여 각 scale에서의 layer 수, channel 수, dimmension 등을 컨트롤합니다. Compound coefficient가 컨트롤하는 정보들은 다음과 같습니다.

$$
W_{bifpn} = 64 * 1.35^\phi \\ 
D_{bifpn} = 3 + \phi \\
D_{box} = D_{class} = 3 + [\phi/3]
R_{input} = 512 + \phi * 128
$$

<p align="center">
  <img src="https://decision-J.github.io/assets/computer_vision/Efficient_Det/compound.png" alt="Compound Scaling"/>
</p>

보시는 것처럼 각 scale별로 자동적으로 coefficient들이 결정되는 것을 볼 수 있습니다. 이를 통해 각 feature들이 조화롭게 학습되기를 바라는 것입니다.


### Experiment Results
Efficient Det이 기존 Object Detection 방법들에 비해 얼마나 좋은 성능을 갖고 있는지 확인해보겠습니다. 

<p align="center">
  <img src="https://decision-J.github.io/assets/computer_vision/Efficient_Det/result.png" alt="Experiment result"/>
</p>

논문에서 다양한 실험 결과들을 확인할 수 있지만 가장 대표적인 것으로 가져왔습니다. 위의 plot을 보시면 모델의 정확도가 다른 기존 방법론들과 비슷한데도 model의 parameter가 가장 작으며, hardware latency도 작음을 알 수 있습니다. 작지만 빠르고 강력한 모델이 바로 Efficient Det이라고 할 수 있겠네요!

### 마무리
지금까지 Efficient Det에 대해 살펴보았습니다. 개인적으로는 parameter 수를 최대한 줄이며 효율성을 추구했는데도 성능이 증가한 부분이 인상 깊습니다. 다만 Compound scaling 부분에서 처럼 hueristic하게 parameter들을 결정해도 모든 케이스에 대해서 같은 성능을 담보할 수 있을지 궁금합니다. ~~(물론 굉장히 유명한 모델인만큼 그럴 것이라 믿어 의심치 않습니다 ㅎㅎ)~~



- - -
### Reference

1. Paper: [Efficient Det](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf)
2. Github: [google/automl/Efficient Det](https://github.com/google/automl/tree/master/efficientdet)
