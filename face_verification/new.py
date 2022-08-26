from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import cv2
import face_recognition
import numpy as np

def facerecog():
    # import libraries

    # Get a reference to webcam 
    video_capture = cv2.VideoCapture(0)

    # Initialize variables
    face_locations = []
    writer = None
    count = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left-5, top-60), (right+5, bottom+5), (0, 0, 255), 2)

        count+=1

        if face_locations!=[] and count%5==0:
            img_path = 'static/frames/frame'+str(count+1)+".jpg"
            cv2.imwrite(img_path,frame)


            result = face_match(img_path,'face_verification/data2.pt')

            if result[1]>1.0:
                print("No match")
                return 'none'

            else:
                return result
                
        if count>30:
            print("No face found")
            return 'none'

def face_match(img_path, data_path, img_vid=False): # img_path= location of photo, data_path= location of data.pt 
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
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

    os.remove(img_path)

    if result[1]>1:
        return "none"
    else:
        return result

if __name__ == '__main__':
    facerecog()
