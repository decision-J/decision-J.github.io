---
layout: post
categories: Competition
title:  "[Kaggle] Google AI Open Images - Object Detection Track"
date:   2021-07-15
author: HaeYong Joung
tags: CV object-detection yolo kaggle
comments: true
---

## Kaggle Study #2. - Google AI Open Images - Object Detection Track

Object Detection에 대한 논문을 읽어가면서 이를 실제 데이터로 적용해보고 싶다는 생각에 과거 kaggle competition을 찾아보았습니다. 그래서 발견한 것이 2018년 진행된  *Google AI Open Images - Object Detection Track* 이었습니다. 3년전 대회이긴 하지만 discussion들이나 코드들을 참고하면서 공부할 목적으로 선택해보았습니다.  

- - -
### 대회 Overview
먼저 *Google AI Open Images - Object Detection Track*에 대해 간단히 정리해보겠습니다.
본 대회는 99,999장의 test 이미지 파일에 대해서 object detection을 수행하는 것이 목적입니다.  

<p align="center">
  <img src="https://decision-J.github.io/assets/kaggle_google_object_detection/overview.png" alt="Example annotations: the house by anita kluska"/>
</p>

Detecting을 제대로 해냈는지 평가하는 지표로 3가지를 살펴봅니다. **Detecting한 object의 category, Box의 좌표, Confidence**입니다. 즉, 이미지 내의 사물을 인식하고, 그 위치를 정확히 포착하고, 이에 대한 예측 확률이 높아야 좋은 rating을 받는 것이죠!

이미지 데이터에 대한 자세한 설명은 [Open Images Challenge page](https://storage.googleapis.com/openimages/web/challenge.html)에서 확인할 수 있습니다!

### How to?
이 문제에 적용해보고 싶은 모델은 YOLO v3입니다.
최근 YOLO 논문들을 version별로 살펴보고 있는데 실제로 얼마나 빠르고 정확한지 확인해보고 싶었거든요!

여러 구현 방법들이 있지만 제가 선택한 방법은 **darknet**을 이용하는 것입니다. 실제 대회 참가가 아니고 우수 코드들을 보면서 공부하는 것이 목적이기 때문에, 우선 제 결과물은 빠르게 내는 것이 중요하다고 생각했습니다. **darknet**은 open source neural network framework로 **C와 CUDA**를 기반으로 짜여져 있어 매우 빠르면서도 코드를 파이썬에서도 쉽게 돌릴 수 있기 때문에 선택하게 되었습니다.

코드는 [AlexeyAB](https://github.com/AlexeyAB/darknet)의 darknet코드 깃헙을 clone하여 사용했습니다. 그리고 darknet을 make하기 전에 제 환경에 맞게 **두 가지 customizing**을 해주었습니다.  

1. **Makefile 수정**

깃헙 코드를 clone하면 그 안에 Makefile이라는 이름의 파일이 생성됩니다. 여러 옵션들을 지정할 수 있는 파일인데요. 저는 *GPU, CUDNN, CUDNN_HALF, OPENCV* 항목을 1로 바꾸어 주었습니다. (기존은 0) 이렇게 해야 GPU를 사용하여 detecting 할 수 있거든요!

2. **detector.c 수정**

src 폴더 내의 detector.c라고 하는 c언어 코드를 일부 수정했습니다. AlexeyAB의 darknet 실행 코드는 몇 가지 옵션을 지정할 수 있습니다. 그 중에서 *save_labels*를 추가하면 detecting한 box의 정보들을 저장해줍니다. 이 때, detecting한 class의 id와 box의 가로 세로 정보가 default로 저장됩니다. 대회의 제출물에 필요한 정보는 class_id, xmin, ymin, xmax, ymax, confidence 값이기 때문에 이 정보들이 저장되도록 코드를 수정했습니다.

**참고**: detector.c를 수정하는 과정에서 *conflicting types for* 오류가 나면서 *detecor.o를 build하는 데 실패*하는 문제가 나타났습니다. 리눅스에서 종종 발생하는 문제라는데.. 이리저리 찾아보면서 헤맸는데 갑작스럽게 저절로 해결되었네요;; 지속적으로 이런 문제가 발생한다면 뭔가 조치가 필요할 것 같습니다.

```C
// pseudo labeling concept - fast.ai
if (save_labels){
    char labelpath[4096];
    replace_image_to_label(input, labelpath);

    FILE* fw = fopen(labelpath, "wb");
    int i;
    for (i = 0; i < nboxes; ++i) {
        char buff[1024];
        int class_id = -1;
        float prob = 0;
        for (j = 0; j < l.classes; ++j) {
            if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                prob = dets[i].prob[j];
                class_id = j;
            }
        }
        if (class_id >= 0) {
            sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f %2.4f\n", class_id,
            prob, #add for confidence
            dets[i].bbox.x - dets[i].bbox.w / 2., #add for xmin
            dets[i].bbox.y - dets[i].bbox.h / 2., #add for ymin
            dets[i].bbox.x + dets[i].bbox.w / 2., #add for xmax
            dets[i].bbox.y + dets[i].bbox.h / 2.); #add for ymax
            fwrite(buff, sizeof(char), strlen(buff), fw);
        }
    }
    fclose(fw);
```

이제 다 됐습니다. 아래 실행 코드를 통해 test 이미지들에 대한 detecting을 수행하기만 하면 됩니다!

```python
!./darknet detector test ./cfg/openimages.data ./cfg/yolov3-openimages.cfg yolov3-openimages.weights -dont_show < ./train.txt > result.txt -save_labels
```

잘 진행되는지 궁금하니까 한 장에 대해서만 수행 결과를 확인해보도록 하겠습니다. 아래 사진은 99,999장의 test 이미지 중 한 장을 가져온 것입니다.

<p align="center">
  <img src="https://decision-J.github.io/assets/kaggle_google_object_detection/test.jfif" alt="test image; 3bc4e7965d04d91f.jpg"/>
</p>

위 사진을 가지고 실행코드를 돌려보면 아래와 같은 실행 결과를 얻을 수 있습니다.

<pre style="auto">
<code>
  /content/darknet
   CUDA-version: 11000 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
   CUDNN_HALF=1
   OpenCV version: 3.2.0
   0 : compute_capability = 600, cudnn_half = 0, GPU: Tesla P100-PCIE-16GB
  net.optimized_memory = 0
  mini_batch = 1, batch = 1, time_steps = 1, train = 0
     layer   filters  size/strd(dil)      input                output
     0 Create CUDA-stream - 0
   Create cudnn-handle 0
  conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 x  32 0.639 BF
     1 conv     64       3 x 3/ 2    608 x 608 x  32 ->  304 x 304 x  64 3.407 BF
     2 conv     32       1 x 1/ 1    304 x 304 x  64 ->  304 x 304 x  32 0.379 BF
     3 conv     64       3 x 3/ 1    304 x 304 x  32 ->  304 x 304 x  64 3.407 BF
     4 Shortcut Layer: 1,  wt = 0, wn = 0, outputs: 304 x 304 x  64 0.006 BF
     5 conv    128       3 x 3/ 2    304 x 304 x  64 ->  152 x 152 x 128 3.407 BF
     6 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
     7 conv    128       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x 128 3.407 BF
     8 Shortcut Layer: 5,  wt = 0, wn = 0, outputs: 152 x 152 x 128 0.003 BF
     9 conv     64       1 x 1/ 1    152 x 152 x 128 ->  152 x 152 x  64 0.379 BF
    10 conv    128       3 x 3/ 1    152 x 152 x  64 ->  152 x 152 x 128 3.407 BF
    11 Shortcut Layer: 8,  wt = 0, wn = 0, outputs: 152 x 152 x 128 0.003 BF
    12 conv    256       3 x 3/ 2    152 x 152 x 128 ->   76 x  76 x 256 3.407 BF
    13 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    14 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    15 Shortcut Layer: 12,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    16 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    17 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    18 Shortcut Layer: 15,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    19 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    20 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    21 Shortcut Layer: 18,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    22 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    23 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    24 Shortcut Layer: 21,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    25 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    26 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    27 Shortcut Layer: 24,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    28 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    29 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    30 Shortcut Layer: 27,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    31 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    32 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    33 Shortcut Layer: 30,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    34 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
    35 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
    36 Shortcut Layer: 33,  wt = 0, wn = 0, outputs:  76 x  76 x 256 0.001 BF
    37 conv    512       3 x 3/ 2     76 x  76 x 256 ->   38 x  38 x 512 3.407 BF
    38 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    39 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    40 Shortcut Layer: 37,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    41 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    42 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    43 Shortcut Layer: 40,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    44 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    45 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    46 Shortcut Layer: 43,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    47 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    48 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    49 Shortcut Layer: 46,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    50 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    51 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    52 Shortcut Layer: 49,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    53 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    54 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    55 Shortcut Layer: 52,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    56 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    57 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    58 Shortcut Layer: 55,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    59 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    60 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    61 Shortcut Layer: 58,  wt = 0, wn = 0, outputs:  38 x  38 x 512 0.001 BF
    62 conv   1024       3 x 3/ 2     38 x  38 x 512 ->   19 x  19 x1024 3.407 BF
    63 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    64 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    65 Shortcut Layer: 62,  wt = 0, wn = 0, outputs:  19 x  19 x1024 0.000 BF
    66 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    67 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    68 Shortcut Layer: 65,  wt = 0, wn = 0, outputs:  19 x  19 x1024 0.000 BF
    69 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    70 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    71 Shortcut Layer: 68,  wt = 0, wn = 0, outputs:  19 x  19 x1024 0.000 BF
    72 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    73 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    74 Shortcut Layer: 71,  wt = 0, wn = 0, outputs:  19 x  19 x1024 0.000 BF
    75 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    76 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    77 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    78 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    79 conv    512       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x 512 0.379 BF
    80 conv   1024       3 x 3/ 1     19 x  19 x 512 ->   19 x  19 x1024 3.407 BF
    81 conv   1818       1 x 1/ 1     19 x  19 x1024 ->   19 x  19 x1818 1.344 BF
    82 yolo
  [yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
    83 route  79 		                           ->   19 x  19 x 512
    84 conv    256       1 x 1/ 1     19 x  19 x 512 ->   19 x  19 x 256 0.095 BF
    85 upsample                 2x    19 x  19 x 256 ->   38 x  38 x 256
    86 route  85 61 	                           ->   38 x  38 x 768
    87 conv    256       1 x 1/ 1     38 x  38 x 768 ->   38 x  38 x 256 0.568 BF
    88 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    89 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    90 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    91 conv    256       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x 256 0.379 BF
    92 conv    512       3 x 3/ 1     38 x  38 x 256 ->   38 x  38 x 512 3.407 BF
    93 conv   1818       1 x 1/ 1     38 x  38 x 512 ->   38 x  38 x1818 2.688 BF
    94 yolo
  [yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
    95 route  91 		                           ->   38 x  38 x 256
    96 conv    128       1 x 1/ 1     38 x  38 x 256 ->   38 x  38 x 128 0.095 BF
    97 upsample                 2x    38 x  38 x 128 ->   76 x  76 x 128
    98 route  97 36 	                           ->   76 x  76 x 384
    99 conv    128       1 x 1/ 1     76 x  76 x 384 ->   76 x  76 x 128 0.568 BF
   100 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
   101 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
   102 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
   103 conv    128       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x 128 0.379 BF
   104 conv    256       3 x 3/ 1     76 x  76 x 128 ->   76 x  76 x 256 3.407 BF
   105 conv   1818       1 x 1/ 1     76 x  76 x 256 ->   76 x  76 x1818 5.376 BF
   106 yolo
  [yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
  Total BFLOPS 148.812
  avg_outputs = 1358830
   Allocate additional workspace_size = 52.43 MB
  Loading weights from yolov3-openimages.weights...
   seen 64, trained: 32013 K-images (500 Kilo-batches_64)
  Done! Loaded 107 layers from weights-file
   Detection layer: 82 - type = 28
   Detection layer: 94 - type = 28
   Detection layer: 106 - type = 28
  ./input/test/3bc4e7965d04d91f.jpg: Predicted in 34.763000 milli-seconds.
  Person: 28%
  Person: 42%
  Man: 32%
  Clothing: 28%
  Person: 49%
  Man: 37%
  Clothing: 41%
  Person: 39%
  Person: 53%
  Man: 30%
  Clothing: 34%
  Person: 35%
  Person: 55%
  Clothing: 29%
</code>
</pre>

상당히 기나 긴 YOLO의 layer들을 거쳐 최종적으로 **Person, Man, Clothing** object들을 찾아낸 것을 확인할 수 있습니다. 글로만 보면 와닿지 않으니 직접 찾아낸 결과들을 눈으로 확인해볼까요?

<p align="center">
  <img src="https://decision-J.github.io/assets/kaggle_google_object_detection/test_result.jfif" alt="Detecting Person, Man, Clothing"/>
</p>

매우 빠른 시간 안에 (거의 3~5초?) 저 class들을 찾아주는 것을 알 수 있습니다. YOLO의 장점인 빠른 속도를 느낄 수 있었습니다.

### Result
<p align="center">
  <img src="https://decision-J.github.io/assets/kaggle_google_object_detection/no_submission.PNG" alt="submission을 만들었는데 왜 제출을 못하니..."/>
</p>

두둥.. submission파일을 열심히 만들고나서야 이걸 깨달았습니다.. 지금 이 대회는 더 이상의 submission을 받지 않고 있네요. 열과 성을 다해 만든 결과물은 아니지만 그래도 제출은 해보고 어느 정도나 rating이 되는지 알고 싶었는데 아쉽습니다 ㅠㅠ


### 마무리
계속되는 변명같지만 코드 공부가 목적이었기 때문에 빠른 시간 안에 baseline 결과물을 만들어 보았습니다.
아무래도 pre-trained model만 가지고 아무런 tuning없이 진행하다보니 성능이 그렇게 좋지는 않습니다. 실제로 위의 result 결과를 보면 사람처럼 <U>큰 class를 제외하고는</U> 잘 찾지 못한 모습을 알 수 있습니다. (컴퓨터, 의자, 책상 등은 detecting 실패) 실제 모델을 적용할 때에는 더 세세한 tuning과 train이 필요할 것으로 보입니다.

- - -
### Reference

1. **Kaggle page**: [Google AI Open Images - Object Detection Track, 2018](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/overview)
2. **Data description**: [Open Images Challenge page](https://storage.googleapis.com/openimages/web/challenge.html)
3. **AlexeyAB Darknet source code**: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
4. **Darknet 활용 참고 블로그**: [https://m.blog.naver.com/bigdata-pro/221781790878](https://m.blog.naver.com/bigdata-pro/221781790878)
5. **Codes**: [decision-j gitgub](https://github.com/decision-J/ComputerVision/tree/main/%5BKaggle%5D%20Google%20AI%20Open%20Images%20-%20Object%20Detection%20Track(2018))
