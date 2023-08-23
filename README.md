# DLforCI
Code for Deep Learning-Based Quantitative Assessment of Renal Chronicity Indices in Lupus Nephritis

## Prerequisites:

We list our environment in requirements.txt

## Segmentation
<code data-enlighter-language="raw" class="EnlighterJSRAW">python segmentation.py --input <img_fold> --model_path <checkpoint_fold> </code>  

Our checkpoints have been uploaded.  

The checkpoints of SAM can be obtained from https://github.com/facebookresearch/segment-anything.  

## CI Assessment
<code data-enlighter-language="raw" class="EnlighterJSRAW">python Calculate_CI.py --input <img_fold> --model_path <checkpoint_fold> </code>  

Here the img_fold represents the folder containing input images that pertain to a single patient's dataset.
