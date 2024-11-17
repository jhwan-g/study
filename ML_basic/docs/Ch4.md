# 로지스틱 회귀
## 분류 문제
- 분류 클래스가 2개: 이진 분류(binary classification)
- 분류 클래스가 여러개: 다중 분류(multi-class classification)

분류 문제에는 K-NN Classifier등 다양한 알고리즘을 사용할 수 있다. 그 중 대표적으로 사용되는 것은 logistic regression이다.

## 로지스틱 회귀
로지스틱 회귀를 통해 이진 분류 문제를 해결할 수 있다.

기존의 선형 회귀는 공역이 R이었음 -> [0, 1]로 제한하고 싶다.  
로지스틱 함수를 사용: $y = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-1(w^Tx + b)}}$  
이 때, y는 x에 대해 class 1일 확률이라고 해석한다(실제 예측 과정에서는 특정한 threshold를 기반으로 0, 1을 결정론적으로 예측함에 주의해야 한다).  
-> $ln(\frac{y}{1-y}) = w^Tx + b$, $\frac{y}{1-y}$를 odds라 하고 거기에 log를 씌운 것을 log odds라고 한다.  
로지스틱 회귀는 주어진 데이터의 log odds에 대한 회귀 직선을 찾는 것으로 생각할 수 있다.

로지스틱 회귀에서 parameter를 구하기 위해선 최대 우도 추정법을 사용할 수 있다.  
데이터 set이 ${(x_i, y_i)}^{m}_{i=1}$이라고 생각할 때  
$log likelihood = l(w, b) = \sum logP(y_i | x_i, w, b) = \sum {((1 - y_i)log y(x) + y_i log (1 - y(x)))}$ 가 성립한다.  
다시 $\hat x_i = (x_i, 1), \beta = (w, b)$로 정의하면 로그 우도를 최대화 하는 것은 $l(\beta) = \sum{(-y_i \beta^T \hat x_i + ln(1 + e^{\beta^T \hat x_i}))}$을 최소화하는 것과 동일함을 알 수 있다.  
이 목표를 달성하기 위해 gradient descent, 뉴턴의 경사 하강법 등을 사용할 수 있다.

# Gradient Descent
## Gradient Descent
경사 하강법은 점진적 학습(온라인 학습)을 가능하게 한다. 점진적 학습은 이전 데이터를 버리지 않고 새로운 데이터에 대한 조정을 가능하게 하는 학습 방법이다.  

경사 하강법은 자주 사용되는 일차 최적화 기법이다. 경사 하강법은 $min_x f(x), f \in C^1 $인 문제를 해결하는 것을 목표로 한다.  
이것을 위해 경사 하강법은 $x_{i + 1} = x_{i} - \gamma \nabla f(x_i)$ 의 규칙으로 값을 업데이트한다.

L-Lipschitz 조건 ($\exist L ~ s.t. ~ \forall x, ~||f(x)|| < L$)이 만족되면 $\gamma = \frac{1}{2L}$일 때 local minimum으로 수렴함이 보장된다는 사실이 알려져 있다.

## Gradient Descent의 여러 종류
경사 하강법에는 여러 종류가 있다. 대표적으로
- Batch Gradient Descent: 한번에 모든 data의 기울기 구해 업데이트
- Minibatch Gradient Descent: 한번에 일부분의 data에 대해 기울기 구해 업데이트
- Stochastic Gradient Descent: 한번에 하나의 data에 대해 기울기 구해 업데이트
훈련 set를 1회 모두 사용하는 것을 1 epoch라고 한다.