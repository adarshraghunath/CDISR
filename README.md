## Info

Inspired from  https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement

### Environment

pip install -r requirement.txt


#### Data Prepare

Data is loaded inside sr.py script

For XRM, prepare Data with dataloader_xrm.py script
For Sinogram, prepare Data with dataloader_sinogram.py script

### Training/Resume Training

python sr_(xrm/sinogram).py -p train -c config/dtsr_(xrm/sinogram).json


### Test/Evaluation


# Edit json to add pretrained model checkpoint path and run the evaluation 
python sr_(xrm/sinogram).py -p val -c config/dtsr_(xrm/sinogram).json  #change the config to add the checkpoint

# Also run measurement_xrm/sinogram to compare with baseline method
python measurement_xrm/sinogram.py

```



