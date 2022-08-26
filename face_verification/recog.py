def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from PIL import Image
    import os

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
    # return (name_list[idx_min], min(dist_list))
    result = (name_list[-1], dist_list[-5])

    os.remove(img_path)

    return result

#     if result[1]>1.0:
#         print("No match")
#     else:
#         print('Face matched with: ',result[0], 'With distance: ',result[1])


# face_match('images//frame51.jpg', 'FaceRec2//pytorch_face_recognition//data.pt')