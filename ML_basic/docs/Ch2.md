# 데이터 다루기
## 데이터 관련 기본 용어

ML - 지도 학습, 비지도 학습, 강화학습 -> 지도 학습은 label(target)이 있지만 비지도 학습은 그렇지 않다

훈련 데이터: 모델의 훈련에 사용되는 데이터  
input: 훈련 데이터에서 모델의 판단 기준이 되는 값 <- feature들을 input으로 사용  
target: 정답 데이터  
sample: 훈련 데이터 하나

## 데이터 나누기

훈련 데이터로 학습 & 평가를 모두 진행하는 것은 적합하지 않다. -> 객관적인 성능 평가가 불가하기 때문이다(과적합 발생 검출 불가).  
- 기본적으로 training set / test set으로 나누어야 하고 보통 7:3 비율을 많이 사용하나 문제에 따라 조정해야 한다.  
- 많은 경우 validation set도 추가해 training set / validation set / test set으로 나눈다.  
-> training set으로 모델을 학습해나가는 과정에서 validation set으로 가장 성능이 잘 나오는 것을 추리고 test set으로 최종 평가를 진행하는 흐름이다.
- sample을 sampling 편향이 나타나지 않도록 잘 나누어야 한다.

## 데이터 전처리

많은 경우 데이터는 적절한 방식으로 전처리(preprocessing)되어야 햔다.  
전처리되지 않은 데이터는 data의 단순한 scale에 영향을 받을 수 있다. 특히 거리 기반 알고리즘의 경우 그 영향이 더 크다.

대표적인 데이터 전처리 방법은 다음과 같다.
- 0-1 Normalization: $z = \frac{x - x_{min}}{x_{max} - x_{min}} $
- Standard Normalization: $z = \frac{x - x_{min}}{\sigma} $

# K-NN 분류기
K-NN 분류기는 인접한 k 개의 원소 중 가장 많은 원소를 가진 class로 분류하는 분류기이다.

N-1인 경우 어떤 지점에서 K-NN 분류기가 잘못 분류할 확률은 가장 가까운 원소와 입력한 원소의 클래스가 다를 확률이며 다음과 같다.  
$P(err) = 1 - \sum_{c}{P(c|x)P(c|z)}$  
z는 x와 가장 가까운 원소, c는 클래스, x는 입력 데이터.
