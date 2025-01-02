# 영상을 활용한 객체 인식2 : 횡단보도 보행자 보호 시스템 프로젝트
### 데이터 설명
- 라벨 : 20종(aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person(보행자),pottedplant,sheep,sofa,train,tvmonitor)
- 입력 데이터 : CCTV 이미지/영상

### 프로젝트 설명
- 필요성 
    - 교차로에서 우회전하는 차량 운전자의 부주의로 인해 횡단보도를 건너는 사람이 다치는 사고가 빈번하게 발생한다. 이를 막기 위해 횡단보도를 촬영하는 CCTV가 우회전하는 차량의 운전자가 확인할 수 있는 위치에 설치된 화면을 통해 **1) 횡단보도**를 보여주고, 동시에 해당 횡단보도에 **2) 보행자가 검출된 경우 사람이 있다고** 화면에 보여주는 방법이 있다.
- 개요

    <img width="200" alt="스크린샷 2024-12-20 오전 12 09 07" src="https://github.com/user-attachments/assets/75ac88c8-9c84-4560-8347-1abb5d58d9eb" />

- 시나리오 설정 
    - 보행자를 감지했을 때 화면으로 "보행자가 있으니 우회전을 금지한다"는 내용 표시
- 모델(가중치) : YOLOv5(Yolov5s), YOLOv7(Yolov7-tiny)

### 파일 구조
- VOCdevkit : Paxcal VOC 데이터세트 폴더
- VOCData : 훈련에 사용할 데이터세트 폴더
- convert2Yolo : label 파일을 YOLOv5 훈련에 사용할 수 있도록 xml을 txt 로 변환할 수 있도록 기능을 포함하는 폴더
- vocnames.txt : 클래스 리스트 파일
- manifest.txt : 이미지 파일 경로 파일
- yolov5(yolov7) : yolov5(yolov7) 실행시 필요한 폴더
    - runs : 훈련 및 추론에 대한 결과가 저장된 폴더
        - train : 훈련 시 생성되는 가중치 및 결과가 저장되는 폴더
        - detect : 추론 시 생성되는 결과 이미지/영상이 저장되는 폴더
    - vocdata.yaml : yolo을 훈련시키기 위한 데이터셋에 대한 정보를 정의한 파일 (클래스 정보, 데이터 경로 포함)

### 결과
- yolov5s

    <img width="200" alt="yolov5s_voc" src="https://github.com/user-attachments/assets/dc45050c-434d-4b5c-8b55-3f65d31d86ce" />

- yolov7-tiny

    <img width="200" alt="yolov7-tiny_voc" src="https://github.com/user-attachments/assets/8bba13e5-0044-40d0-99a9-54973edf8787" />
