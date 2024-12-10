# -*- coding: utf-8 -*-
"""Datatset_Class.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12hoBJzr73pfjquzREphEd-kZ6siAe5L2
"""

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class PyTorch_Custom_Dataset_Class(Dataset):
  def __init__(self):
    super().__init__()
    pass

  def __getitem__(self,idx):
    pass

  def __len__(self):
    pass


class PyTorch_Classification_Dataset_Class(Dataset):
  def __init__(self,dataset_dir="./Recycle_Classification_Dataset",transform=None):
    super().__init__()
    # 데이터세트 다운로드
    if not os.path.isdir(dataset_dir):
      os.system("git clone https://github.com/JinFree/Recycle_Classification_Dataset.git")
      os.system("rm -rf ./Recycle_Classification_Dataset/.git")
    self.image_abs_path=dataset_dir

    # 전처리방법
    self.transform=transform
    if self.transform is None:
      self.transform=transforms.Compose([
          transforms.Resize(256), # 256*256으로 크기 조정
          transforms.RandomCrop(224), # 랜덤하게 224*224 영역 추출
          transforms.ToTensor(), # 0~255 => 0~1로 변환
          transforms.Normalize( # 정규화
              mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]
        )
    ])

    # 클래스 이름 리스트 정렬
    self.label_list=os.listdir(self.image_abs_path)
    self.label_list.sort()
    self.x_list=[]
    self.y_list=[]

    # 폴더 내 모든 파일의 경로 리스트 제작
    for label_index,label_str in enumerate(self.label_list):
      img_path=os.path.join(self.image_abs_path,label_str)
      img_list=os.listdir(img_path)
      for img in img_list:
        self.x_list.append(os.path.join(img_path,img))
        self.y_list.append(label_index)

  def __len__(self):
    return len(self.x_list)
    
  def __getitem__(self,idx): # 해당 인덱스(idx)의 이미지와 클래스 정보 반환
    image=Image.open(self.x_list[idx])

    # 흑백 이미지를 위한 예외 처리
    if image.mode != 'RGB':
      image=image.convert('RGB')
    if self.transform is not None: # 전처리가 안됐다면 전처리 수행
      image=self.transform(image)
    return image,self.y_list[idx]

  # 클래스 이름 리스트 저장
  def __save_label_map__(self,dst_text_path='label_map.txt'):
    label_list=self.label_list
    f=open(dst_text_path,'w')
    for i in range(len(label_list)):
      f.write(label_list[i]+'\n')
    f.close()

  # 학습 과정에 사용하기 위해 전체 클래스의 수 출력
  def __num_classes__(self):
    return len(self.label_list)

