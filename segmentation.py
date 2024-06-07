import argparse
import logging
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 80).__str__()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segmentation_model1.deeplabv3 import DeepLabV3Plus
from segmentation_model1.res_classifier import Res_classifier
from segmentation_model1.unet import UNet
from sklearn.metrics import matthews_corrcoef
from tools import *


def segment(
    in_files, BCI_glomeruli, BCI_tubule, GLI_gs, GLI_fc, IFTA_identifier, mask_generator
):

    for i, fn in enumerate(os.listdir(in_files)):
        try:
            img = cv2.imread(in_files + "/" + fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(fn, ": check you image file")

        glomeruli_mask = segment_glomeruli(img, BCI_glomeruli, mask_generator)

        tubule_mask = segment_tubule(img, BCI_tubule, mask_generator)

        glomeruli_mask[glomeruli_mask > 0] = 1
        glomeruli_mask = glomeruli_mask.astype(np.uint8)
        glomeruli_mask = dilate_demo(glomeruli_mask)
        glomeruli_mask[glomeruli_mask > 0] = 1
        glomeruli_mask, _ = identify_gs_fc(glomeruli_mask, img, GLI_gs, GLI_fc)

        print("get  glomeruli_mask and tubule_mask")
        IF_TA_mask = segment_IFTA(img, IFTA_identifier, glomeruli_mask, tubule_mask)
        IF_TA_mask[glomeruli_mask > 0] = 0
        print("get IFTA mask")

        glomeruli_mask = cv2.resize(
            glomeruli_mask, (IF_TA_mask.shape[1], IF_TA_mask.shape[0])
        )

        tubule_mask = cv2.resize(
            tubule_mask, (IF_TA_mask.shape[1], IF_TA_mask.shape[0])
        )
        full_img = cv2.resize(img, (IF_TA_mask.shape[1], IF_TA_mask.shape[0]))
        tubule_mask[tubule_mask > 0] = 4

        glomeruli_mask = mask_to_image(glomeruli_mask)

        glomeruli_mask = add_mask_img(full_img, glomeruli_mask)
        cv2.imwrite("./vis/glomeruli_mask_" + fn, glomeruli_mask)

        tubule_mask = mask_to_image(tubule_mask)

        tubule_mask = add_mask_img(full_img, tubule_mask)
        cv2.imwrite("./vis/tubule_mask_" + fn, tubule_mask)

        IF_TA_mask = mask_to_image(IF_TA_mask)

        IF_TA_mask = add_mask_img(full_img, IF_TA_mask)
        cv2.imwrite("./vis/IF_TA_mask_" + fn, IF_TA_mask)


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="filenames of input images",
        default="./imgs",
    )

    parser.add_argument(
        "--model_path", "-p", help="Scale factor for the input images", default="./ckpt"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    model_path = args.model_path
    epoch = args.model_epoch

    sam_checkpoint = model_path + "sam_vit_h_4b8939.pth"

    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.cuda()

    mask_generator = SamAutomaticMaskGenerator(sam)

    BCI_glomeruli = UNet(3, 2)
    BCI_glomeruli.load_state_dict(torch.load(model_path + "/BCI_glomeruli.pth"))
    BCI_glomeruli = BCI_glomeruli.cuda()

    BCI_tubule = UNet(3, 2)
    BCI_tubule.load_state_dict(torch.load(model_path + "/BCI_tubule.pth"))
    BCI_tubule = BCI_tubule.cuda()

    GLI_gs = Res_classifier(
        encoder_name="resnet18", encoder_weights="imagenet", classes=1
    )
    GLI_gs.load_state_dict(torch.load(model_path + "/GLI_gs.pth"))
    GLI_gs = GLI_gs.cuda()

    GLI_fc = Res_classifier(
        encoder_name="resnet18", encoder_weights="imagenet", classes=1
    )
    GLI_fc.load_state_dict(torch.load(model_path + "/GLI_fc.pth"))
    GLI_fc = GLI_fc.cuda()

    IFTA_identifier = DeepLabV3Plus(encoder_name="resnet50", classes=2)
    IFTA_identifier.load_state_dict(torch.load(model_path + "/IFTA_identifier.pth"))
    IFTA_identifier = IFTA_identifier.cuda()

    logging.info("Model loaded !")

    segment(
        in_files,
        BCI_glomeruli,
        BCI_tubule,
        GLI_gs,
        GLI_fc,
        IFTA_identifier,
        mask_generator,
    )
