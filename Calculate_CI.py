import argparse
import logging

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,80).__str__()
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from segmentation_model1.unet import UNet
from segmentation_model1.deeplabv3 import DeepLabV3Plus
from segmentation_model1.res_classifier import Res_classifier
from sklearn.metrics import matthews_corrcoef

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from tools import *

def Calculate_CI(in_files,BCI_glomeruli,BCI_tubule,GLI_gs,GLI_fc,IFTA_identifier,mask_generator):
   
    # images belongs to the same patient
    try:
        img = cv2.imread (in_files)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print(in_files,": check you image file") 
    
    glomeruli_mask=segment_glomeruli(img,BCI_glomeruli,mask_generator)

    tubule_mask=segment_tubule(img,BCI_tubule,mask_generator)


    glomeruli_mask[glomeruli_mask>0]=1
    glomeruli_mask=glomeruli_mask.astype(np.uint8)
    glomeruli_mask=dilate_demo(glomeruli_mask)
    glomeruli_mask[glomeruli_mask>0]=1
    glomeruli_mask,class_set=identify_gs_fc(glomeruli_mask,img,GLI_gs,GLI_fc)
    all_number,seg,cre=cal_glomeruli(class_set)
   

    print(f"GS: {seg/all_number}; FC: {cre/all_number}")

    IF_TA_mask=segment_IFTA(img,IFTA_identifier,glomeruli_mask,tubule_mask)
    IF_TA_mask[glomeruli_mask>0]=0
   
    ifta_area,all_area=cal_if_ci(glomeruli_mask,img,IF_TA_mask)
        
    ta_area,all_t_area=cal_ta_ci(IF_TA_mask,img)


    print(f"IF: {ifta_area/all_area}; TA: {ta_area/all_t_area}")
        
   





def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images beongs to the same patient', default='./imgs')

    parser.add_argument('--model_path', '-p', 
                        help="Scale factor for the input images",
                        default='./ckpt')

    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    model_path=args.model_path
    epoch=args.model_epoch

    sam_checkpoint = model_path+"sam_vit_h_4b8939.pth"

    
    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.cuda()

    mask_generator = SamAutomaticMaskGenerator(sam)
  
    BCI_glomeruli =UNet(3,2)
    BCI_glomeruli.load_state_dict(torch.load(model_path+'/BCI_glomeruli.pth'))
    BCI_glomeruli=BCI_glomeruli.cuda()
    #new_state_dict = OrderedDict()

    BCI_tubule =UNet(3,2)
    BCI_tubule.load_state_dict(torch.load(model_path+'/BCI_tubule.pth'))
    BCI_tubule=BCI_tubule.cuda()

    GLI_gs=Res_classifier(encoder_name='resnet18',encoder_weights='imagenet',classes=1)
    GLI_gs.load_state_dict(torch.load(model_path+'/GLI_gs.pth'))
    GLI_gs=GLI_gs.cuda()

    GLI_fc=Res_classifier(encoder_name='resnet18',encoder_weights='imagenet',classes=1)
    GLI_fc.load_state_dict(torch.load(model_path+'/GLI_fc.pth'))
    GLI_fc=GLI_fc.cuda()

    IFTA_identifier =DeepLabV3Plus(encoder_name = "resnet50",classes=2)
    IFTA_identifier.load_state_dict(torch.load(model_path+'/IFTA_identifier.pth'))
    IFTA_identifier=IFTA_identifier.cuda()

  
   
    logging.info("Model loaded !")

    Calculate_CI(in_files,BCI_glomeruli,BCI_tubule,GLI_gs,GLI_fc,IFTA_identifier,mask_generator)
   
        