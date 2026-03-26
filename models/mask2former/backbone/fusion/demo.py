import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from utils.tools import *


if __name__ == "__main__":
    img = np.array(Image.open('/media/ubuntu/Data/autopet/data_all/PETCT_0af7ffe12a/08-12-2005/CT/48.tif').convert('RGB'))
    img = normalize(img)
    plt.imshow(img)
    plt.show()

    img = torch.from_numpy(img.transpose(2, 0, 1))
    print(img.shape)
    image_vis_ycrcb = RGB2YCrCb(img[None]).cpu().numpy()[0].transpose(1, 2, 0)
    print(image_vis_ycrcb.shape)

    plt.imshow(image_vis_ycrcb[..., 2])
    plt.show()