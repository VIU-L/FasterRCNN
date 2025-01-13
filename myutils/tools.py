def CenterSize_to_TwoPoints(center_x,center_y,height,width):
    xmin=center_x-width/2
    xmax=center_x+width/2
    ymin=center_y-height/2
    ymax=center_y+height/2
    return xmin,ymin,xmax,ymax
def TwoPoints_to_CenterSize(xmin,ymin,xmax,ymax):
    center_x=(xmin+xmax)/2
    center_y=(ymin+ymax)/2
    width=xmax-xmin
    height=ymax-ymin
    return center_x,center_y,height,width