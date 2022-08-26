from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import cv2
import numpy as np
from face_verification.yolo_detection_webcam import facedetector

def facerecog():
    face_location = 0
    writer = None
    count = 0

    while True:
        face_locations = facedetector()
        print("In")
        if face_locations:
            img_path = 'static/frames/frame.jpg'

            result = face_match(img_path,'face_verification/data2.pt')

            if result=='none' or result[1]>1.0:
                print("No match")
                return 'none'

            else:
                return result
                

def face_match(img_path, data_path, img_vid=False): # img_path= location of photo, data_path= location of data.pt 
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    try:
        emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
        
        saved_data = torch.load(data_path) # loading data.pt file
        embedding_list = saved_data[0] # getting embedding data
        name_list = saved_data[1] # getting list of names
        dist_list = [] # list of matched distances, minimum distance is used to identify the person
        
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
            
        idx_min = dist_list.index(min(dist_list))
        result = (name_list[idx_min], min(dist_list))

        print(result[0],result[1])
        os.remove(img_path)

        if result[1]>1:
            return "none"
        else:
            return result
    
    except:
        print("none")
        os.remove(img_path)
        return "none"

if __name__ == '__main__':
    facerecog()
