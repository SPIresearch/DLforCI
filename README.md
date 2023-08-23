# DLforCI
Code for Deep Learning-Based Quantitative Assessment of Renal Chronicity Indices in Lupus Nephritis

## Prerequisites:

We list our environment in requirements.txt.

Please place the images, which can either be whole-slide images (WSI) or cropped patches depending on your available computer resources, into an "img_fold."

Additionally, ensure that the checkpoint files are situated within the "checkpoint_fold." Kindly note that our checkpoints have been uploaded for your use. For the SAM checkpoints, you can access the following URL: https://github.com/facebookresearch/segment-anything.

## Segmentation
<code data-enlighter-language="raw" class="EnlighterJSRAW">python segmentation.py --input <img_fold> --model_path <checkpoint_fold> </code>  

## CI Assessment
<code data-enlighter-language="raw" class="EnlighterJSRAW">python calculate_CI.py --input <img_fold> --model_path <checkpoint_fold> </code>  

Note that here the img_fold should be the folder containing input images that pertain to a single patient's dataset.
