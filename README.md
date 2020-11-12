---

[1. CNN을 통한 수화 판별](#cnn-을-통한-수화-판별)

[2. CNN 모델](###cnn-모델)

[3. 검증과 예측 정확도](###검증과-예측-정확도)

---



# CNN 을 통한 수화 판별

##### 학습 데이터

- A ~ Z 까지 각 1400장 총 36400장의 이미지
- 50*50 픽셀
- RGB

##### 검증 데이터

- A ~ Z 까지 각 100장 총 2600장의 이미지
- 50*50 픽셀
- RGB

[CNN 모델 및 npy 파일 확인](https://www.dropbox.com/sh/pagfd3a32a8y1ro/AADLPFJoAuqJHSgdt-T5k8kPa?dl=0)

---

### CNN 모델

- CNN 모델 학습 과정 (loss 그래프)

  ![CNN 학습 loss 그래프](https://user-images.githubusercontent.com/68371545/98892601-0631c300-24e4-11eb-9726-ce74ffcf415f.png)

- CNN 모델 시각화

![CNN 시각화](https://user-images.githubusercontent.com/68371545/98892677-295c7280-24e4-11eb-94b2-b721b2010464.png)

- 모델 summary

![CNN 모델 summary](https://user-images.githubusercontent.com/68371545/98892714-3b3e1580-24e4-11eb-9d81-b98773741076.png)

---

### 검증과 예측 정확도

![CNN결과](https://user-images.githubusercontent.com/68371545/98892803-6a548700-24e4-11eb-868a-26b5ae3f663b.JPG)



