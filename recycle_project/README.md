# 이미지 분류를 활용한 재활용품 분류 : 재활용품 분리수거 프로젝트
### 데이터 설명
- 라벨 : 캔, 유리병, 플라스틱, 종이
- 입력 데이터 : 컨베이어 벨트 위의 쓰레기 이미지/영상
### 프로젝트 설명
- 시나리오 설정 : 한 번에 하나의 쓰레기만 촬영하여 재활용 종류와 정확도 화면에 표시
- 모델 : DNN

### 파일 구조
- Dataset_Class.py : 데이터 세트 클래스
  - Pytorch_Custom_Dataset_Class(Dataset)
    - __init__(self)
    - __getitem__(self)
    - __len__(self)
  - Pytorch_Classification_Datatset_Class(Dataset)
    - __init__(self,dataset_dir="/content/Recycle_Classification_Dataset",transform=None) : 데이터 세트 다운로드 및 전처리, 경로설정
    - __getitem__(self,idx) : # 해당 인덱스(idx)의 이미지와 클래스 정보 반환
    - __len__(self)
    - __save_label_map__(self,dst_text_path='label_map.txt') : 클래스 이름 리스트 저장
    - __num_class__(self) : 학습 과정에 사용하기 위해 전체 클래스의 수 출력
- Model_Class_From_the_Scratch.py : 심층 신경망 직접 구현
  - PyTorch_Custom_Model_Class(nn.Module)
    - __init__(self)
    - forward(self,x)
  - MODEL_From_Scratch(nn.Module)
    - forward(self,x)
    - __init__(self,num_classes)
- Model_Class_Transfer_Learning_MobileNet.py : 전이학습을 활용한 심층 신경망 구현
  - MobileNet(nn.Module)
    - __init__(self,num_classes,pretrained=True)
    - forward(self,x)
- Training_Class.py : 학습 클래스
  - PyTroch_Classification_Training_Class()
    - __init__(
      self,
      dataset_dir='/content/Recycle_Classification_Dataset',
      batch_size=16,
      train_ratio=0.75
      )
    - prepare_network(self,is_scratch=True) : 직접 구현 / 전이학습 신경망에 따른 신경망 구조 초기화
    - training_network(self) : 하이퍼파라미터 설정 및 실제 훈련 수행
- PyTorch_Recycle_Classification_Colab.ipynb : 심층 신경망 학습 파일
- Inference_Image.ipynb : 이미지 추론 파일

- Inference_Cam.py : 재활용품 분류 추론



### 결과
- 직접 구현 : 69m 39s / Accuracy: 61.61%

  <img width="189" alt="스크린샷 2024-12-22 오후 9 31 27" src="https://github.com/user-attachments/assets/e300021e-365e-4c4e-a87a-c113300aff4c" />
  
- 전이학습 : 340m 42.3s / Accuracy: 95.67%

  <img width="201" alt="스크린샷 2024-12-22 오후 9 31 35" src="https://github.com/user-attachments/assets/9be4e1d0-7a9a-4b68-8138-c4480f125501" />
