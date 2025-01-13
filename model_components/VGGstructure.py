import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool
from RPN import RPN
from anchors import *
from ROItools import *

import sys
sys.path.append("C:/Users/zyr/Desktop/Master/练习2/FasterRCNN")
from myutils.tools import *

class myVGG(nn.Module):
    def __init__(self,RPN=RPN,use_local_weights=False):
        super(myVGG, self).__init__()
        pretrained_vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # input (1,3,H,W)
        self.feature_extractor = nn.Sequential(
            *list(pretrained_vgg16.features.children())[:-1]) # all conv layers
        # now (1,3,H/16,W/16)
        # todo: during training, freeze layers 0-9
        # todo: gaussianize new layers' weights
        
        self.RPN = RPN() # todo
        self.roi_pooling = RoIPool(output_size=(7, 7), spatial_scale=1/16) # {boxes},all harmonized to 512,7,7
        
        self.FClayers=nn.Sequential(
            *list(pretrained_vgg16.classifier.children())[:-1]
        )
        self.cls_head=nn.Linear(4096,21)# 20 types + 1 background
        self.reg_head=nn.Linear(4096,84)# corresponding box of the 21 classes'
        # Initialize weights of custom layers to N(0, 0.01)
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.reg_head.weight, mean=0.0, std=0.01)
        if use_local_weights:
            local_weights_path = "model_components/goodeweights/vgg16.pth"
            self.load_state_dict(torch.load(local_weights_path))
    def forward(self, x):
        features = self.feature_extractor(x)
        self.RPN(features)
        self.RPN.propose()
        self.proposed_boxes=self.RPN.assembly_proposed_pieces()

        # [(center_x,center_y,width,height),...]
        y=self.roi_pooling(features,[self.proposed_boxes])
        # now (N,512,7,7)
        y = y.view(y.size(0), -1)
        # now (N,25088)
        y=self.FClayers(y)
        # now (N,21)
        cls=self.cls_head(y)
        self.cls=torch.softmax(cls,dim=1)
        self.regs=self.reg_head(y)
        self.regs=self.regs.view(-1,21,4)
        # now (N,21,4)
        return self.cls,self.regs
    def decide(self):
        absolute_original_boxes,predicted_labels=get_absolute_original_boxes(self.regs,self.cls,self.proposed_boxes)
        return absolute_original_boxes,predicted_labels
        
    def annotate_and_save(self,idx_of_image,TrainOrTest='Train'):
        pass
            

if __name__ == "__main__":
    print("start to read")
    IMG,old_size,truth=read_single_data('000298','Train')
    input_tensor=IMG.to(device)
    print("input loaded")
    vgg=myVGG().to(device)
    cls,regs=vgg(input_tensor)
    print("forwarded")
    absolute_original_boxes,predicted_labels=get_absolute_original_boxes(regs,cls,vgg.proposed_boxes)
    best_match_boxes,best_match_labels=get_ROI_truth(absolute_original_boxes,truth)
    loss=calculate_ROI_loss(best_match_boxes,best_match_labels,cls,regs)
    print(loss)
    
    # # Example input tensor
    # IMG,old_size,truth=read_single_data('000298','Train')
    # input_tensor = IMG.to(device)  # Batch size = 1, Channels = 3
    # minibatch_anchors,minibatch_feature_anchors,minibatch_boxesAndClasses=generate_minibatch_from_image(IMG,truth)
    
    # vgg=myVGG().to(device)
    # cls,regs=vgg(input_tensor)
    # print(cls.shape,regs.shape)
    # decisions=vgg.decide()
    # print(decisions)
    
    # features=vgg.feature_extractor(input_tensor)
    # posneg,regress=vgg.RPN(features)
    # loss=calculate_RPNloss(posneg,regress,minibatch_boxesAndClasses,minibatch_feature_anchors)
    # print(loss)
    