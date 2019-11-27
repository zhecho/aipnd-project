
from PIL import Image
import numpy as np


def process_image(image):
    """ Scales, crops and normalizes PIL image for PyTorch model

    :image: path to image
    :returns: returns numpy array

    """
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((16,16,240,240))
    img = np.asarray(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image - mean / std
    img = (img - mean)/std
    img_t = img.transpose(2,0,1)
    
    return img_t

