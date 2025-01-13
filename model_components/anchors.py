from torchvision.ops import box_iou,nms
import torch
import sys
sys.path.append("C:/Users/zyr/Desktop/Master/练习2/FasterRCNN")
from myutils.tools import CenterSize_to_TwoPoints
from myutils.dataparse import read_single_data
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

anchor_sizes=torch.Tensor([
    (91, 181),
    (181, 362),
    (362, 724),
    (128, 128),
    (256, 256),
    (512, 512),
    (181, 91),
    (362, 181),
    (724, 362)
]).to(device)# these are sizes on Original Picture!


def generate_all_anchors(H, W): # thank you GPT
    # Define grid center offsets, considering a stride of 16 and center offset of 8
    stride = 16
    offset = stride // 2

    # Create a grid of center coordinates
    center_x = torch.arange(offset, H, stride, device=device)
    center_y = torch.arange(offset, W, stride, device=device)
    grid_x, grid_y = torch.meshgrid(center_x, center_y, indexing='ij')

    # Flatten the grid
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    # Repeat grid for each anchor size
    num_anchors = 9
    num_grid_points = grid_x.size(0)
    
    grid_x = grid_x.repeat(9)
    grid_y = grid_y.repeat(9)
    
    anchor_widths = anchor_sizes[:, 1].repeat_interleave(num_grid_points)
    anchor_heights = anchor_sizes[:, 0].repeat_interleave(num_grid_points)

    # Compute anchor box corners
    xmin = grid_x - anchor_widths / 2
    ymin = grid_y - anchor_heights / 2
    xmax = grid_x + anchor_widths / 2
    ymax = grid_y + anchor_heights / 2

    # Create mask to filter out-of-bound anchors
    valid_mask = (xmin >= 0) & (ymin >= 0) & (xmax <= H) & (ymax <= W)

    # Filter anchors and corresponding feature indices
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze()

    all_anchors = torch.stack([xmin, ymin, xmax, ymax], dim=1)[valid_indices]
    corr_feature_space_info=torch.stack([grid_x//16,grid_y//16,torch.arange(9).repeat(num_grid_points).to(device)],dim=1)[valid_indices]


    return all_anchors.to(device), corr_feature_space_info.to(device)


    
def true_PosNegTag_list(all_anchors,truth): 
    # check for paper to see if we don't need this to train ROI
    tags=torch.zeros((all_anchors.shape[0],5),dtype=torch.float).to(device)
    true_boxes=[obj['new_bounding_box'] for obj in truth]
    true_boxes=torch.tensor(true_boxes).to(device)
    BoxLabels=[obj['class'] for obj in truth]
    BoxLabels=torch.tensor(BoxLabels).to(device)

    
    ious=box_iou(all_anchors,true_boxes)
    max_ious, _ = ious.max(dim=1)
    tags[max_ious > 0.5] = getRelativeBoxes(all_anchors[max_ious > 0.5], true_boxes[ious[max_ious > 0.5].argmax(dim=1)],BoxLabels[ious[max_ious > 0.5].argmax(dim=1)])
    tags[max_ious < 0.2] = torch.Tensor([-1, -1, -1, -1,-1]).to(device)
    _, max_iou_indices = ious.max(dim=0)
    tags[max_iou_indices] = getRelativeBoxes(all_anchors[max_iou_indices], true_boxes,BoxLabels)
    return tags
    # todo: sneak in the class labels for VGG training
        
def random_anchor_minibatch(all_anchors,truth):# we want 128 pos 128 neg
    tags=true_PosNegTag_list(all_anchors,truth)
    # shape: (N,5)
    pos_indices = torch.nonzero(tags[:, 0] > 0, as_tuple=False).squeeze()
    neg_indices = torch.nonzero(tags[:, 0] == -1, as_tuple=False).squeeze()
    L=len(pos_indices)
    if L>128:
        pos_indices=torch.randperm(L)[:128].to(device)
        neg_indices=torch.randperm(len(neg_indices))[:128].to(device)
        all_indices=torch.cat((pos_indices,neg_indices),0)
    else:
        neg_indices=torch.randperm(len(neg_indices))[:256-L].to(device)
        all_indices=torch.cat((pos_indices,neg_indices),0)
    all_boxesAndClasses=tags[all_indices,:]
    return all_indices,all_boxesAndClasses
def generate_minibatch_from_image(image,truth):
    now_size=image.shape[2:]
    all_anchors,corr_featureanchors=generate_all_anchors(now_size[0],now_size[1])
    minibatch_indices,minibatch_boxesAndClasses=random_anchor_minibatch(all_anchors,truth)
    minibatch_anchors=all_anchors[minibatch_indices]
    minibatch_feature_anchors=corr_featureanchors[minibatch_indices]
    return minibatch_anchors,minibatch_feature_anchors,minibatch_boxesAndClasses


def getRelativeBoxes(Anchors,Boxes,BoxLabels):
    # Anchors,Boxes: (N,4)
    x1a,y1a,x2a,y2a=Anchors[:,0],Anchors[:,1],Anchors[:,2],Anchors[:,3]
    x1b,y1b,x2b,y2b=Boxes[:,0],Boxes[:,1],Boxes[:,2],Boxes[:,3]
    wa,ha=x2a-x1a,y2a-y1a
    wb,hb=x2b-x1b,y2b-y1b
    xa,ya=(x1a+x2a)/2,(y1a+y2a)/2
    xb,yb=(x1b+x2b)/2,(y1b+y2b)/2
    tx=(xb-xa)/wa
    ty=(yb-ya)/ha
    tw=torch.log(wb/wa)
    th=torch.log(hb/ha)
    return torch.stack([tx,ty,tw,th,BoxLabels],dim=1)


def calculate_RPNloss(pos_neg,regressions,minibatch_labels,minibatch_feature_anchors):
    # Unpack minibatch_feature_anchors into x, y, k (in a vectorized way)
    x, y, k = minibatch_feature_anchors[:, 0], minibatch_feature_anchors[:, 1], minibatch_feature_anchors[:, 2]
    
    # Extract the corresponding regression values and labels
    relative_boxes = minibatch_labels[:, :4]# last position is class label
    labels = (relative_boxes[:, 0] > 0).long()  # Convert the boolean to an integer (0 or 1)
    # Get the confidence scores from pos_neg
    confidence_of_positive= pos_neg[0, k, 1, x, y]
    bce_loss = F.binary_cross_entropy(confidence_of_positive, labels.float().to(device),reduction='mean')
    
    # Mask for labels == 1 (only consider regressions where the label is 1)
    positive_mask = labels == 1

    # Perform smooth L1 loss for positive labels
    this_regressions = regressions[0, k, :, x, y]
    smooth_l1_loss = F.smooth_l1_loss(this_regressions[positive_mask], relative_boxes[positive_mask], reduction='mean')

    # Total loss is the sum of BCE loss and Smooth L1 loss
    total_loss = bce_loss + 10*smooth_l1_loss
    print("cls loss:",bce_loss.item(),"regression loss:",smooth_l1_loss.item())
    return total_loss
def calculate_ROIloss(cls,regs,minibatch_labels,minibatch_feature_anchors):
    # note that tims time minibatch_feature_anchors should explicitly contain
    # the proposal shape instead of anchor type
    pass
            

# some utils for eval
def get_feature_space_box(window_center_feature,relative_box,original_anchor_size):
    if type(original_anchor_size)==int:
        anchor_type=original_anchor_size
        original_anchor_size=anchor_sizes[anchor_type]# if original_anchor_size is a k
    wa=original_anchor_size[0]/16
    ha=original_anchor_size[1]/16
    xa,ya=window_center_feature
    
    tx,ty,tw,th=relative_box
    x=xa+tx*wa
    y=ya+ty*ha
    w=wa*torch.exp(tw)
    h=ha*torch.exp(th)
    return torch.Tensor(CenterSize_to_TwoPoints(x,y,w,h))
def get_feature_space_boxes_batch(window_centers_feature,relative_boxes,original_anchor_size,size_is_feature=False):
    if size_is_feature:
        original_anchor_size=original_anchor_size*16
    if type(original_anchor_size)==int:
        anchor_type=original_anchor_size
        original_anchor_size=torch.Tensor(anchor_sizes[anchor_type])# if original_anchor_size is a k
    wa=original_anchor_size[0]/16
    ha=original_anchor_size[1]/16
    xa,ya=window_centers_feature[:,0],window_centers_feature[:,1]
    tx, ty, tw, th = relative_boxes[0], relative_boxes[1], relative_boxes[2], relative_boxes[3]
    x=xa+tx*wa
    y=ya+ty*ha
    w=wa*torch.exp(tw)
    h=ha*torch.exp(th)
    x1s,y1s,x2s,y2s=CenterSize_to_TwoPoints(x,y,w,h)
    return torch.stack([x1s,y1s,x2s,y2s],dim=1)# shape: (N,4)
def get_original_space_box(feature_space_box):
    return 16*feature_space_box
def purge(scores,boxes):    
    # scores = torch.tensor([proposal[2] for proposal in proposals])
    # boxes = torch.stack([get_feature_space_box(proposal[0], proposal[1], proposal[3]) for proposal in proposals])
    final_proposals=nms(boxes,scores, 0.7)# indices of proposals to keep, in descending order of scores
    if len(final_proposals)>300:final_proposals=final_proposals[:300]# keep top 300
    return boxes[final_proposals]
if __name__ == "__main__":
    # IMG,old_size,truth=read_single_data('000298','Train')
    # minibatch_anchors,minibatch_feature_anchors, minibatch_labels=generate_minibatch_from_image(IMG,truth)
    # print(minibatch_anchors[0],minibatch_labels[0])
    # print(minibatch_feature_anchors[0])
    
    pass