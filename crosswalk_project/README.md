# 영상을 활용한 객체 인식2 : 횡단보도 보행자 보호 시스템 프로젝트
### 데이터 설명
- 라벨 : 보행자
- 입력 데이터 : CCTV 영상

### 프로젝트 설명
- 필요성 
    - 교차로에서 우회전하는 차량 운전자의 부주의로 인해 횡단보도를 건너는 사람이 다치는 사고가 빈번하게 발생한다. 이를 막기 위해 횡단보도를 촬영하는 CCTV가 우회전하는 차량의 운전자가 확인할 수 있는 위치에 설치된 화면을 통해 **1) 횡단보도**를 보여주고, 동시에 해당 횡단보도에 **2) 보행자가 검출된 경우 사람이 있다고** 화면에 보여주는 방법이 있다.
- 개요

    <img width="200" alt="스크린샷 2024-12-20 오전 12 09 07" src="https://github.com/user-attachments/assets/75ac88c8-9c84-4560-8347-1abb5d58d9eb" />

- 시나리오 설정 
    - 보행자를 감지했을 때 화면으로 "보행자가 있으니 우회전을 금지한다"는 내용 표시

- 모델 : YOLOv5s, YOLOv7-tiny

### 결과
- yolov5s

    <img width="200" alt="yolov5s_voc" src="https://github.com/user-attachments/assets/dc45050c-434d-4b5c-8b55-3f65d31d86ce" />

- yolov7-tiny

    <img width="200" alt="yolov7-tiny_voc" src="https://github.com/user-attachments/assets/8bba13e5-0044-40d0-99a9-54973edf8787" />
