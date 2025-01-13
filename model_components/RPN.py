import torch
import torch.nn as nn
import torch.nn.functional as F
from anchors import *

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        # input 1,512,H/16,W/16
        self.slidingWindow_layer = nn.Conv2d(512, 256, kernel_size=3,stride= 1, padding=1)
        # add relu
        # now 1,256,H/16,W/16
        self.classification_layer = nn.Conv2d(256, 9*2, kernel_size=1,stride= 1, padding=0) 
        # this gives 2 scores (pos,neg) for each anchor
        # size 1,18,H/16,W/16
        self.regression_layer = nn.Conv2d(256, 9*4, kernel_size=1,stride= 1, padding=0)
        # this gives 9 proposal predictions for each point, each is (tx,ty,tw,th) defined in page 4
    def forward(self, x):
        x = F.relu(self.slidingWindow_layer(x))
        classifications = self.classification_layer(x)
        pos_neg=classifications.view(1,9,2,classifications.shape[2],classifications.shape[3])
        pos_neg=F.softmax(pos_neg,dim=2)
        # now 1,9,2(neg;pos),H/16,W/16
        regressions = self.regression_layer(x)
        regressions = regressions.view(1,9,4,regressions.shape[2],regressions.shape[3])
        # now 1,9,4(tx,ty,tw,th),H/16,W/16
        self.PosNeg=pos_neg
        self.regressions=regressions
        return self.PosNeg,regressions
    def propose(self):
        all_scores=torch.Tensor([]).to(device)
        all_boxes=torch.Tensor([]).to(device)
        for k in range(9):
            anchor_size=anchor_sizes[k]
            scores=self.PosNeg[0,k,1,:,:].view(-1)# list-like
            relative_boxes=self.regressions[0,k,:,:,:].view(4,-1) # 4,N
            window_centers_feature=torch.tensor([(i,j) for i in range(self.PosNeg.shape[3]) for j in range(self.PosNeg.shape[4])]).to(device)
            
            boxes=get_feature_space_boxes_batch(window_centers_feature,relative_boxes,anchor_size)# already in 2-point format
            all_scores=torch.cat((all_scores,scores),0)
            all_boxes=torch.cat((all_boxes,boxes),0)
                    
        # during eval, proposals are always in feature space!
        self.scores=all_scores
        self.boxes=all_boxes
        return all_scores,all_boxes
    def assembly_proposed_pieces(self):# during eval
        return purge(self.scores,self.boxes)
    
if __name__ == "__main__":

    # Example input tensor
    input_tensor = torch.randn(1, 3, 600, 727)  # Batch size = 1, Channels = 3, H = W = 224

    output =input_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    # 'output' shape: (batch_size, channels * kernel_size^2, output_height * output_width)
    print(output.shape)
