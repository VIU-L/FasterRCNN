# %%
import cv2
import os
import pickle
import torch
import xml.etree.ElementTree as ET
import sys
sys.path.append("C:/Users/zyr/Desktop/Master/练习2/FasterRCNN")
from myutils.tools import *
from myutils.classNameMap import *

train_image_folder = 'TrainData/JPEGImages/'
train_annotation_folder = 'TrainData/Annotations/'

test_image_folder = 'TestData/JPEGImages/'
test_annotation_folder = 'TestData/Annotations/'

def parse_image(image_path,new_short_side=600):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width = image.shape[:2]
    old_size=(width,height)
    if height < width:
        new_height = new_short_side
        new_width = int((new_short_side / height) * width)
        ratio=new_short_side/height
    else:
        new_width = new_short_side
        new_height = int((new_short_side / width) * height)
        ratio=new_short_side/width
    image = cv2.resize(image, (new_width, new_height))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return image, old_size,ratio

def parse_xml(annotation_path,ratio):
    xml_data= open(annotation_path).read()
    root = ET.fromstring(xml_data)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        objects.append({"class": CLASSNAMES_MAP[name], "bounding_box": (xmin,ymin,xmax,ymax),"new_bounding_box":(xmin*ratio,ymin*ratio,xmax*ratio,ymax*ratio)})
    return objects
    
    
def read_single_data(idx,TrainOrTest='Train'):
    if TrainOrTest=='Train':
        image_folder=train_image_folder
        annotation_folder=train_annotation_folder
    else:
        image_folder=test_image_folder
        annotation_folder=test_annotation_folder
    filename=idx+'.jpg'
    image_path = os.path.join(image_folder, filename)
    IMG,old_size,ratio=parse_image(image_path)

    annotation_path = os.path.join(annotation_folder, idx + '.xml')
    truth=parse_xml(annotation_path,ratio)
    return IMG, old_size,truth
    
def create_list_available(TrainOrTest='Train'):
    if TrainOrTest=='Train':
        image_folder=train_image_folder
    else:
        image_folder=test_image_folder
    list_available=[]
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            list_available.append(filename.split('.')[0])
    with open(f'{TrainOrTest}Data/availables.txt', 'w') as f:
        for item in list_available:
            f.write("%s\n" % item)
def get_list_available(TrainOrTest='Train'):
    list_available=[]
    with open(f'{TrainOrTest}Data/availables.txt', 'r') as f:
        for line in f:
            list_available.append(line.strip())
    return list_available
   
# %%
if __name__ == '__main__':
    IMG,old_size,truth=read_single_data('000005','Train')
    print(IMG,old_size,truth)
    print(IMG.shape)
    
    
