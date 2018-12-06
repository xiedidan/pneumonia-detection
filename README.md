# pneumonia-detection
## Mask RCNN
Best result (single network, w/o verifier):  
```
mAP: 0.16288442463458827
hit:	121
miss:	26
fp:	108
fn:	81
neg:	664
```
Config:  
``` python
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet101'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_PADDING = False

    RPN_NMS_THRESHOLD  = 0.9
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    TRAIN_ROIS_PER_IMAGE = 16

    USE_MINI_MASK = False
    
    MAX_GT_INSTANCES = 5
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.3
```
Training with config ```RPN_ANCHOR_RATIOS = [0.25, 0.33, 0.5, 1, 2, 3, 4]``` is still in progress.  
