# :loop: 줄넘기 개수 측정 프로그램(개인 프로젝트) :loop:

<br/>

## :pushpin: 개요
   - 프로그램 소개
   - 개발 동기
   - 기술 스택
   - 프로그램 흐름도
   - 후기

---
<br/>

## ✔️ 프로그램 소개
줄넘기 개수 측정 프로그램은 인공지능 기술 기반으로 사용자의 줄넘기 수행 영상을 입력으로 받아 프레임별 각 관절 좌표 데이터를 이용하여 줄넘기 수행 개수를 측정하는 프로그램입니다.

---

<br/>

## ✔️ 개발 동기
줄넘기 운동은 빠른 템포와 호흡을 필요로 하기 때문에 개인이 목표 개수를 정해두어도 운동을 하다 보면 스스로 넘은 개수를 세는 것에 어려움을 겪곤 합니다. 이러한 어려움을 해결하기 위해 대부분은 스마트 워치를 이용해 사용자의 움직임을 감지하여 줄넘기를 넘은 개수를 측정하거나, 줄넘기를 돌리면 자동으로 개수가 측ㄹ정되는 카운트 줄넘기를 사용하여 줄넘기를 넘은 개수를 측정합니다. 하지만 스마트 워치를 이용하는 방식은 비교적 고가의 스마트 워치를 소유해야만 사용할 수 있기 때문에 이용 가능 대상이 한정적이라는 단점이 존재하고, 카운트 줄넘기는 개수 측정의 정확도가 많이 떨어진다는 단점이 존재합니다.

줄넘기 개수 측정 프로그램은 스마트폰 등을 이용하여 사용자의 줄넘기 수행 영상을 녹화한 후 인공지능 모델을 이용하여 줄넘기 수행 개수를 측정하는 방식으로 스마트 워치를 이용하여 줄넘기 수행 개수를 측정하는 방식보다 이용 가능 대상자의 범위가 넓고, 카운트 줄넘기를 이용하여 줄넘기 수행 개수를 측정하는 방식보다 더 높은 정확도를 가진다는 장점이 있습니다.


---

<br/>

## :shopping_cart: 기술 스택
- 프로그래밍 언어
   - Python

- 프레임 워크 및 주요 라이브러리
   - Tensorflow
   - Openpose
   - OpenCV
     
- 사용한 인공지능 모델
   - Gradient Boosting Classifier
   - KNeighbors Classifier
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Classifier
   - DNN
   - LSTM

---

<br/>

## ✔️ Flow Chart
- 프로그램의 전체적인 흐름도
- 사용자의 줄넘기 수행영상을 입력으로 받아 openpose 라이브러리에서 제공하는 MPII 모델을 통해 각 프레임별 관절 좌표 데이터를 추출
- 추출한 관절 좌표데이터를 학습 데이터셋으로 사용하여 5개의 머신러닝 모델과 2개의 딥러닝 모델을 학습.
<br/>

![프로그램 개요도](https://github.com/KJirung/JumpRope_Count_Measurement/assets/142071404/15df2634-8e8b-4e52-916d-8cf7d94318f6)


---

<br/>

## ✔️ 후기
처음으로 데이터셋 없이 진행한 프로젝트로 데이터셋 생성 및 수집부터 모델 학습까지 혼자 수행하면서 프로젝트 일련의 과정을 이해할 수 있었습니다. 또한 오즈비 분석, 특성 중요도 분석, Confusion Matrix 분석 등 데이터 분석에 사용되는 다양한 기법들을 활용해봄으로써 데이터를 분석하는 실력을 한층 더 키울 수 있었던 프로젝트였습니다.

