# 전처리
import cv2
from PIL import Image
import torchvision.transforms as transforms

# 추론
import torch
from Model_Class_From_the_Scratch import MODEL_From_Scratch
from Model_Class_Transfer_Learning_MobileNet import MobileNet

# 후처리
import numpy as np
import argparse # 어떤 신경망 사용할 것인지, 어떤 동영상 스트림을 추론할 것인지 결정

class Inference_Class():
    def __init__(self): # 초기화
        USE_CUDA=torch.backends.mps.is_available()
        self.DEVICE=torch.device('mps' if USE_CUDA else "cpu")
        self.model=None
        self.label_map=None
        self.transform_info=transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()
        ])
        
    def load_model(self,is_train_from_scratch,label_map_file='label_map.txt'): # 신경망과 가중치, 클래스 이름 리스트 호출
        self.label_map=np.loadtxt(label_map_file,str,delimiter='\t')
        num_classes=len(self.label_map)
        model_str=None
        if is_train_from_scratch:
            self.model=MODEL_From_Scratch(num_classes).to(self.DEVICE)
            model_str='PyTorch_Training_From_Scratch'
        else:
            self.model=MobileNet(num_classes).to(self.DEVICE)
            model_str='PyTorch_Transfer_Learning_MobileNet'
        model_str+='.pt'
        self.model.load_state_dict(torch.load(model_str,map_location=self.DEVICE))
        
    def inference_video(self,video_source='test_video.mp4'): # 영상 추론
        cap=cv2.VideoCapture(video_source)
        if cap.isOpened(): # 정상적으로 비디오가 열리는지 확인
            print('Video Opened')
        else:
            print('Video Not Opened')
            print('Program Abort')
            exit() # 실행 중지
        cv2.namedWindow('Output',cv2.WINDOW_GUI_EXPANDED) #창 생성 및 추론 결과 영상 제시
        
        with torch.no_grad(): # 자동미분연산(=autograd)엔진을 비활성화 => 메모리 사용량 감소, 순전파 속도 증가
            while cap.isOpened():
                ret,frame=cap.read() # 프레임을 정상적으로 받아왔는지의 값, 실제 동영상 프레임 반환
                if ret:
                    output=self.inference_frame(frame) # 프레임 추론
                    cv2.imshow('Output',output)
                else:
                    break
                if cv2.waitKey(33)& 0xFF ==ord('q'): # 30FPS이므로 33ms동안 동영상 제시 , 'q'입력 시 종료
                    break
            cap.release()
            cv2.destroyAllWindows()
        return
    
    def inference_image(self, opencv_image):
        opencv_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(opencv_rgb)
        image_tensor = self.transform_info(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.DEVICE)
        with torch.no_grad():
            inference_result = self.model(image_tensor)
        inference_result = inference_result.squeeze()
        inference_result = inference_result.cpu().numpy()
        result_frame = np.copy(opencv_image)
        label_text = self.label_map[np.argmax(inference_result)]
        class_prob = str(inference_result[np.argmax(inference_result)])
        result_frame = cv2.putText(result_frame, label_text + " " + class_prob, (10, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=(0,0,255), thickness=3)
        return result_frame, label_text, class_prob    
                
    def inference_frame(self,opencv_frame):
        opencv_rgb=cv2.cvtColor(opencv_frame,cv2.COLOR_BGR2RGB) # 색공간 변환
        image=Image.fromarray(opencv_rgb)
        image_tensor=self.transform_info(image)
        image_tensor=image_tensor.unsqueeze(0) # 4차원 텐서로 변환 (기존 3차원 이미지 데이터에 1차원 추가)
        image_tensor=image_tensor.to(self.DEVICE) # gpu로 데이터 이동
        
        inference_result=self.model(image_tensor) # 모델 실행
        inference_result=inference_result.squeeze() # 차원 줄이기
        inference_result=inference_result.cpu().numpy()  # 각 클래스에 대한 확률
        
        result_frame=np.copy(opencv_frame)
        label_text=self.label_map[np.argmax(inference_result)]
        label_text+=" "+str(inference_result[np.argmax(inference_result)])
        result_frame=cv2.putText(
            result_frame,label_text,(10,50),cv2.FONT_HERSHEY_PLAIN, fontScale=2.0,color=(0,0,255),thickness=3
        )
        return result_frame
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "-s",'--is_scratch',required=False,action='store_true',
        help='inference with model trained from the scratch') 
    # "-s",'--is_scratch' : 단축형 -s 또는 전체 이름 --is_scratch 중 하나를 사용하여 옵션을 활성화
    # required=False : 명령줄에서 반드시 입력할 필요가 없다는 것
    # action='store_true' : 해당 옵션이 사용되면, 해당 변수의 값이 True로 설정된다는 것
    # help : 옵션의 설명 텍스트
    parser.add_argument(
        "-src",'--source',required=False, type=str,default='./test_video.mp4',
        help='OpenCV Video source'
    )
    args=parser.parse_args()
    is_train_from_scratch=False
    source=args.source
    if args.is_scratch:
        is_train_from_scratch=True
    inferenceClass=Inference_Class() # 선언 및 초기화
    inferenceClass.load_model(is_train_from_scratch) # 신경망 불러오기
    inferenceClass.inference_video(source) # 추론 실행