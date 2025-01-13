from torchvision.ops import box_iou,nms
import torch
import sys
sys.path.append("C:/Users/zyr/Desktop/Master/练习2/FasterRCNN")
from myutils.tools import *
from myutils.dataparse import read_single_data
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from anchors import *
from VGGstructure import myVGG

def get_absolute_original_boxes(regs,cls,proposals_on_feature):
    # regs: (N,21,4), containing (dx,dy,dw,dh) for each class
    # cls: (N,21), containing the softmaxed scores for each class
    # proposal_on_feature: a (N,4) tensor of (x1,y1,x2,y2) on feature map
    # return: a tensor of (x1,y1,x2,y2) on original image
    N=cls.shape[0]
    predicted_labels=torch.argmax(cls,dim=1)
    regs_of_predicted_class=regs[torch.arange(N),predicted_labels]
    # shape (N,4)
    x1_proposal_feature, y1_proposal_feature, x2_proposal_feature, y2_proposal_feature = proposals_on_feature.t()
    
    xp,yp,wp,hp=(x1_proposal_feature+x2_proposal_feature)/2,(y1_proposal_feature+y2_proposal_feature)/2,x2_proposal_feature-x1_proposal_feature,y2_proposal_feature-y1_proposal_feature
    
    tx,ty,tw,th=regs_of_predicted_class[:,0],regs_of_predicted_class[:,1],regs_of_predicted_class[:,2],regs_of_predicted_class[:,3]
    
    x=tx*wp+xp
    y=ty*hp+yp
    w=torch.exp(tw)*wp
    h=torch.exp(th)*hp
    
    absolute_feature_box=torch.stack([x-w/2,y-h/2,x+w/2,y+h/2],dim=1)
    absolute_original_boxes=absolute_feature_box*16
    return absolute_original_boxes,predicted_labels
    
def get_ROI_truth(absolute_original_box,truth):
    # absolute_original_box:(N,4)
    true_boxes=[obj['new_bounding_box'] for obj in truth]
    true_boxes=torch.tensor(true_boxes).to(device)
    true_labels=[obj['class'] for obj in truth]
    true_labels=torch.tensor(true_labels).to(device)
    
    ious=box_iou(absolute_original_box, true_boxes)
    best_match_indexes=torch.argmax(ious,dim=1)
    
    best_match_boxes=true_boxes[best_match_indexes]
    best_match_labels=true_labels[best_match_indexes]
    best_match_labels[ious.max(dim=1).values < 0.3] = 20 # do we need this?
    return best_match_boxes, best_match_labels
    
def calculate_ROI_loss(best_match_boxes,best_match_labels,cls,regs):
    # best_match_boxes:(N,4)
    # best_match_labels:(N,) # containing classes 0~20
    # cls:(N,21)
    # regs:(N,21,4)
    N=cls.shape[0]
    cls_loss=F.cross_entropy(cls,best_match_labels,reduction='mean')
    
    valid_mask = best_match_labels != 20 # not empty
    
    regs_loss=F.smooth_l1_loss(regs[torch.arange(N),best_match_labels][valid_mask], best_match_boxes[valid_mask], reduction='mean')
    total_loss=cls_loss+10*regs_loss
    print(cls_loss.item(),regs_loss.item())
    return total_loss
