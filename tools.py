import argparse
import logging
import pdb,copy
from operator import ne
import albumentations as A
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,80).__str__()
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from segmentation_model1.unet import UNet
from segmentation_model1.deeplabv3 import DeepLabV3Plus
from segmentation_model1.res_classifier import Res_classifier
from sklearn.metrics import matthews_corrcoef
from collections import OrderedDict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from iou_score import IoUScore

test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
def form_list(y_list,x_list):
    point_set=[]
    for y in y_list:
        for x in x_list:
            point_set.append([y,x])
    return point_set

def segment_IFTA(img,IFTA_identifier,glomeruli_mask,tubule_mask):
    mask_ifta = identify_IFTA_region(net=IFTA_identifier,
                            full_img=img,
                            nclasses=2,scale_factor=4,weight=[0.1,0.9])
    mask_ifta=mask_ifta.astype(np.uint8)
    mask_ifta=dilate_demo(mask_ifta,10)
    IF_TA_mask = distinguish_IF_TA(
                            full_img=img,
                            tubule_mask=tubule_mask,mask_ifta=mask_ifta,patch_size=2048,glomeruli_mask=glomeruli_mask,nclasses=3,scale_factor=2)
  
   
    IF_TA_mask=cv2.resize(IF_TA_mask,(mask_ifta.shape[1],mask_ifta.shape[0]),interpolation=cv2.INTER_AREA)
    return IF_TA_mask
def identify_IFTA_region(net,
                full_img,
                
                scale_factor=4,
                nclasses=2,patch_size=1024,weight=[1,1,1.3]):
    full_img=cv2.resize(full_img,(full_img.shape[1]//scale_factor,full_img.shape[0]//scale_factor))
   
 
    h_p, w_p = patch_size-full_img.shape[0]%patch_size, patch_size-full_img.shape[1]%patch_size
    full_img = cv2.copyMakeBorder(full_img, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
   
    mask_image=np.zeros([nclasses,full_img.shape[0],full_img.shape[1]])

    img_h,img_w,_=full_img.shape
    step=int(0.5*patch_size)
    num_h=img_h//step
    num_w=img_w//step
    y_list=[i*step for i in range(0,num_h)]
    x_list=[i*step for i in range(0,num_w)]
    point_set=form_list(y_list,x_list)
    net.eval()
  
    for y,x in point_set:
        if (x+patch_size)>img_w or (y+patch_size)>img_h:
            continue
       
        img=full_img[y:y+patch_size,x:x+patch_size]#.transpose(2,1,0)
        
        img=test_transform(image=img)['image']
        
        img = torch.from_numpy(img.transpose(2, 0, 1))#)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output= net(img)
            
        
            probs = F.softmax(output, dim=1)
           
            probs = probs.squeeze(0)
            full_mask = probs.squeeze().cpu().numpy()
            for i in range(nclasses):
                full_mask[i,:,:]*=weight[i]
            mask_image[:,y:y+patch_size,x:x+patch_size]=mask_image[:,y:y+patch_size,x:x+patch_size]+full_mask

            
    mask_image=np.argmax(mask_image,axis =0)
    full_img=full_img[:-h_p,w_p:,:]
    mask_image=mask_image[:-h_p,w_p:]
    return mask_image
def distinguish_IF_TA(
                full_img,
                tubule_mask,mask_ifta,glomeruli_mask,mask_generator,
                scale_factor=4,
                nclasses=2,patch_size=2048):
    full_img=cv2.resize(full_img,(full_img.shape[1]//scale_factor,full_img.shape[0]//scale_factor))
    mask_ifta=mask_ifta.astype(np.uint8)
    mask_ifta=cv2.resize(mask_ifta,(full_img.shape[1],full_img.shape[0]))
    tubule_mask=tubule_mask.astype(np.uint8)
    tubule_mask=cv2.resize(tubule_mask,(full_img.shape[1],full_img.shape[0]))

    h_p, w_p = patch_size-full_img.shape[0]%patch_size, patch_size-full_img.shape[1]%patch_size
    full_img = cv2.copyMakeBorder(full_img, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
    glomeruli_mask= cv2.copyMakeBorder(glomeruli_mask, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
    tubule_mask= cv2.copyMakeBorder(tubule_mask, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
    mask_ifta= cv2.copyMakeBorder(mask_ifta, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
    mask_image=np.zeros([full_img.shape[0],full_img.shape[1]])

    img_h,img_w,_=full_img.shape
    step=int(patch_size)
    num_h=img_h//step
    num_w=img_w//step
    y_list=[i*step for i in range(0,num_h)]
    x_list=[i*step for i in range(0,num_w)]
    point_set=form_list(y_list,x_list)
   
    z=0
    for y,x in point_set:
        z+=1
    
        img=full_img[y:y+patch_size,x:x+patch_size]
        if_area=mask_ifta[y:y+patch_size,x:x+patch_size]
        sub_glomeruli=glomeruli_mask[y:y+patch_size,x:x+patch_size]
        sub_tubule=tubule_mask[y:y+patch_size,x:x+patch_size]
        sam_mask=mask_generator.generate(img)
        print(z,len(point_set))
    

        img=test_transform(image=img)['image']
 
        img = torch.from_numpy(img.transpose(2, 0, 1))#)
        img = img.unsqueeze(0)
        img = img.cuda()
      

        full_mask=if_area
        full_mask[full_mask>0]=6

        if (sub_tubule *if_area).sum()>100:
            full_mask[sub_tubule==1]=5
             

        mask_image[y:y+patch_size,x:x+patch_size]=full_mask
        #k
   
    mask_image=mask_image.astype(np.uint8)#np.argmax(mask_image,axis =0)
    full_img=full_img[:-h_p,w_p:,:]
    mask_image=mask_image[:-h_p,w_p:]
    return mask_image


def got_label(mask,v=[]):
    for i in range(len(v)):
        mask[mask==i]=v[i]
    return mask    
def cal_glomeruli_area(mask):
    mask_glomeruli=copy.deepcopy(mask)
    mask_glomeruli=got_label(mask_glomeruli,[0,1,1,1,1,1,0,0,0])
    return mask_glomeruli

def cal_glomeruli(class_set):
    all_number=len(class_set)
    class_set=np.array(class_set)
  
    seg=(class_set==1).sum()#got_stat((mask==1)+1)
    cre=(class_set==3).sum()#got_stat((mask==3)+1)
    return all_number,seg,cre

def cal_if(mask):
   
    return (mask==6).sum()

def cal_ta(mask):
   
    
    return (mask==5).sum()

def cal_nt_ta(mask):
   

    return (mask==1).sum()

def cal_ta_ci(tubule_mask,img,IF_TA_mask):
    ta=cal_ta(IF_TA_mask)
    nt_ta=cal_nt_ta(tubule_mask)
    return ta,nt_ta

def cal_if_ci(glomeruli_mask,img,IF_TA_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_the_whole=the_whole_area(gray)
    mask_the_whole=cv2.resize(mask_the_whole,(IF_TA_mask.shape[1],IF_TA_mask.shape[0]),interpolation=cv2.INTER_AREA)
   
    ifta=cal_if(IF_TA_mask)
    all=(mask_the_whole==1).sum()-(cal_glomeruli_area(glomeruli_mask)==1).sum()
    # 
    return ifta,all

label2color_d={'background': [0, 0, 0], 'GS':[255, 255, 0],'other_glomeruli':[0, 0, 255], 
'FC':[0, 255, 0], 'normal_tubules':[255, 0, 128], 
'TA':[255,0,0], 'IF':[255,235,205]}


label2color=[]
for k,v in label2color_d.items():
    label2color.append(v[::-1])

#label2color=[[0, 0, 0],[160,32,240],[255, 255, 255]]
label2color=np.array(label2color)



def mask_to_image(mask):
    img=np.zeros([mask.shape[0],mask.shape[1],3], dtype=np.uint8)

    mask=mask.astype(int)
    img=np.array(label2color)[mask]
    return img
 


def add_mask_img(img,mask):
    mask1=np.sum(mask,2,keepdims=True)
    mask1[mask1>0]=1
    
    img1=img*(1-mask1)
    img=(img*0.3 + mask*0.7)+img1*0.7#.transpose(1, 2, 0)
    img=cv2.resize(img*1.0,(img.shape[1]//2+1,img.shape[0]//2+1)).astype(np.int)
    return img#Image.fromarray((mask * 255).astype(np.uint8))



def dilate_demo(image,k=5):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dst = cv2.dilate(image, kernel)

    return dst
def erode_demo(image,k=11):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    dst = cv2.erode(image,kernel)
    return dst
def filter_connectedComponents(mask_glomeruli):
    
    ccNum, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_glomeruli.astype( np.uint8 ))
    #pdb.set_trace()
    k=0
    
    for i in range(ccNum):
        #print(stats[i][-1])
        if stats[i][-1]<3600:
            mask_glomeruli[labels==i]=0
    
    return mask_glomeruli


def the_whole_area(gray,counter_thre=30):
    img_h,img_w=gray.shape
    gray=cv2.resize(gray,(img_w//4,img_h//4))
    detected_edges = cv2.blur(gray,(3,3),0)
    
    while True:
        ret, binary = cv2.threshold(detected_edges, counter_thre, 255, cv2.THRESH_BINARY)
        counter_thre+=30
        #print(counter_thre)
        if binary.sum()<gray.shape[0]*gray.shape[1]*255*4/5:
            break
        if counter_thre>230:
            break
    
    detected_edges = 255-binary
 
    dst=detected_edges
  
    h = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    contours = h[0]
    #import pdb; 
    temp = np.ones(dst.shape,np.uint8)*255
    #画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
    cv2.drawContours(temp,contours,-1,(0,255,0),cv2.FILLED)
    
    
    ccNum, labels, stats, centroids = cv2.connectedComponentsWithStats(temp.astype( np.uint8 ))
    #import pdb; 
    notallow=[]
    labels=labels+1
    for k in  range(len(stats)):
        if stats[k][4]<400:
            #notallow.append(k)
            labels[labels==(k+1)]=0
    #for k in notallow

    labels[labels>0]=1
    #import pdb; 
    temp1=(temp*labels).astype(np.uint8)
    temp1[temp1>0]=1
    
    temp1=cv2.resize(temp1,(img_w,img_h),interpolation=cv2.INTER_AREA)
    temp1=1-temp1
    return temp1




def predict_img_glomeruli(net,
                full_img,mask_generator,
                sam_refine=False,
                scale_factor=4,
                nclasses=2,patch_size=1024,weight=[1,1,1.3]):
    full_img=cv2.resize(full_img,(full_img.shape[1]//scale_factor,full_img.shape[0]//scale_factor))

    h_p, w_p = patch_size-full_img.shape[0]%patch_size, patch_size-full_img.shape[1]%patch_size
    full_img = cv2.copyMakeBorder(full_img, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
   
    if nclasses > 1:
        mask_image=np.zeros([nclasses,full_img.shape[0],full_img.shape[1]])
    else:
        mask_image=np.zeros([full_img.shape[0],full_img.shape[1]])
   
    img_h,img_w,_=full_img.shape
    step=int(0.3*patch_size)
    num_h=img_h//step
    num_w=img_w//step
    y_list=[i*step for i in range(0,num_h)]
    x_list=[i*step for i in range(0,num_w)]
    point_set=form_list(y_list,x_list)
    net.eval()
  
    for y,x in point_set:
      
        if (x+patch_size)>img_w or (y+patch_size)>img_h:
            continue
       
        img=full_img[y:y+patch_size,x:x+patch_size]#.transpose(2,1,0)
        if sam_refine:
            sam_mask=mask_generator.generate(img)
        
        img=test_transform(image=img)['image']
       


        img = torch.from_numpy(img.transpose(2, 0, 1))#)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output,f = net(img)
          
            probs = F.softmax(output, dim=1)
          
            probs = probs.squeeze(0)
            full_mask = probs.squeeze().cpu().numpy()
            for i in range(nclasses):
                full_mask[i,:,:]*=weight[i]
            full_mask=np.argmax(full_mask,axis =0)

        if sam_refine:
            max_iou_score=0
            small_mask=full_mask*0
            for s_mask in sam_mask:
            
                sub_mask=s_mask['segmentation'].astype(np.uint8)
                area=s_mask['area']
                iou_score=IoUScore(2)(area,full_mask)
                if iou_score>max_iou_score and iou_score>0.6:
                    full_mask*=0
                    full_mask[sub_mask==1]=1
                    max_iou_score=iou_score
                elif area>=img.shape[2]*img.shape[3]/100: #for small tubules
                    small_mask[sub_mask==1]=1
            full_mask=small_mask+full_mask
        mask_image[:,y:y+patch_size,x:x+patch_size]=mask_image[:,y:y+patch_size,x:x+patch_size]+full_mask         
    mask_image[mask_image>0]=1
    full_img=full_img[:-h_p,w_p:,:]
    mask_image=mask_image[:-h_p,w_p:]

    return mask_image

def segment_glomeruli(img,BCI_glomeruli,mask_generator):
    mask_glomeruli= predict_img_glomeruli(net=BCI_glomeruli,
                            full_img=img,mask_generator=mask_generator,
                            nclasses=2,scale_factor=4,weight=[1,1.3])
    ccNum, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_glomeruli.astype( np.uint8 ))
    
    for i in range(ccNum):
        #print(stats[i][-1])
        if stats[i][-1]<3600:
            mask_glomeruli[labels==i]=0
    
    return mask_glomeruli

def get_ord(mask):
    lists=[]
    mask[mask>0]=255
    ccNum, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype( np.uint8 ))
    for i in range(ccNum):
        if stats[i][-1]>mask.shape[0]*mask.shape[1]/4 or stats[i][-1]<400:
            continue
        # if mask.sum()<255*100:
        #     continue
        x,y,w,h,_=stats[i]
        if x==0 and y==0:
            continue
        x=max(0,x-50)
        y=max(0,y-50)
        w=w+100
        h=h+100
        lists.append([x,y,w,h,i])
    return labels,lists

def identify_gs_fc(mask,img,GLI_gs,GLI_fc):
    mask=cv2.resize(mask,(mask.shape[1]*4,mask.shape[0]*4),interpolation=cv2.INTER_NEAREST)
    img=cv2.resize(img,(mask.shape[1],mask.shape[0]))
    #print(img.shape,mask.shape)
    mask[mask>0]=1
    mask=mask*255
    l_mask=np.zeros(mask.shape)
    labels,ll=get_ord(mask.astype(np.uint8))
    
    GLI_gs=GLI_gs.eval()
    GLI_fc=GLI_fc.eval()
    with torch.no_grad():   
        class_set=[]
        for x,y,w,h,i in ll:
            #pdb.set_trace()
            
            img1=img[y:y+h,x:x+w]#.transpose(2, 0, 1)
            img1=test_transform(img1)
            
            #img1 = torch.from_numpy(img1.transpose(2, 0, 1))#)
            img1 = img1.unsqueeze(0)
            img1 = img1.cuda()
            outputs_s,_=GLI_gs(img1)
            outputs_s= torch.sigmoid(outputs_s)
            pred_s=(outputs_s>0.9)+0
            pred_s=pred_s.squeeze()
            outputs_f,_=GLI_fc(img1)
            outputs_f= torch.sigmoid(outputs_f)
            pred_f=(outputs_f>0.01)+0
            pred_f=pred_f.squeeze()
            
            if pred_f==1:
                img_class=3
            elif pred_s==0:
                img_class=1
            else:
                img_class=2

            class_set.append(img_class)

            l_mask[labels==i]=img_class
           
                
    return l_mask,class_set


def segment_tubule(
                full_img,net,mask_generator,
                sam_refine=False,
                scale_factor=4,
                nclasses=2,patch_size=1024,weight=[1,1,1.3]):
    full_img=cv2.resize(full_img,(full_img.shape[1]//scale_factor,full_img.shape[0]//scale_factor))
    
    h_p, w_p = patch_size-full_img.shape[0]%patch_size, patch_size-full_img.shape[1]%patch_size
    full_img = cv2.copyMakeBorder(full_img, 0,h_p, w_p,0, cv2.BORDER_CONSTANT, value=0)
   
   
    mask_image=np.zeros([full_img.shape[0],full_img.shape[1]])
    
    img_h,img_w,_=full_img.shape
    step=int(0.3*patch_size)
    num_h=img_h//step
    num_w=img_w//step
    y_list=[i*step for i in range(0,num_h)]
    x_list=[i*step for i in range(0,num_w)]
    point_set=form_list(y_list,x_list)
    net.eval()
  
    for y,x in point_set:
      
        if (x+patch_size)>img_w or (y+patch_size)>img_h:
            continue
       
        img=full_img[y:y+patch_size,x:x+patch_size]#.transpose(2,1,0)
        if sam_refine:
            sam_mask=mask_generator.generate(img)
        
        img=test_transform(image=img)['image']
       


        img = torch.from_numpy(img.transpose(2, 0, 1))#)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output,f = net(img)
          
            probs = F.softmax(output, dim=1)
          
            probs = probs.squeeze(0)
            full_mask = probs.squeeze().cpu().numpy()
            for i in range(nclasses):
                full_mask[i,:,:]*=weight[i]
            full_mask=np.argmax(full_mask,axis =0)

        if sam_refine:
            max_iou_score=0
            small_mask=full_mask*0
            for s_mask in sam_mask:
            
                sub_mask=s_mask['segmentation'].astype(np.uint8)
                area=s_mask['area']
                iou_score=IoUScore(2)(area,full_mask)
                if iou_score>max_iou_score and iou_score>0.8:
                    full_mask*=0
                    full_mask[sub_mask==1]=1
                    max_iou_score=iou_score
                elif area<=img.shape[2]*img.shape[3]/100: #for small tubules
                    small_mask[sub_mask==1]=1
            full_mask=full_mask+small_mask          
        mask_image[:,y:y+patch_size,x:x+patch_size]=mask_image[:,y:y+patch_size,x:x+patch_size]+full_mask         
    mask_image[mask_image>0]=1
    full_img=full_img[:-h_p,w_p:,:]
    mask_image=mask_image[:-h_p,w_p:]
    

    return mask_image


def got_label(mask,v=[]):
    for i in range(len(v)):
        mask[mask==i]=v[i]
    return mask
