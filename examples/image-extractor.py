import os

import matplotlib.pyplot as plt
import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader, Dataset

from habitat_sim.utils.data.data_extractor import ImageExtractor


# Helper functions
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]
    label = sample["label"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


# Give the extractor a path to the scene
scene_filepath = "/media/shubodh/DATA/Downloads/data-non-onedrive/replica_v1/apartment_0/habitat/mesh_semantic.ply"

# Instantiate an extractor. The only required argument is the scene filepath
extractor = ImageExtractor(
    scene_filepath,
    labels=[0.0],
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
)

# Index in to the extractor like a normal python list
sample = extractor[0]
print("length of extractor or number of images")
print(len(extractor))
# Or use slicing
samples = extractor[1:4]
for sample in samples:
    display_sample(sample)

