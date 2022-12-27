# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz
# 
# Question: how does the undersegmentation error (ue) depend on the number of superpixels?
# Answer: The higher the number of superpixels the low is the undersegmentation error.
# When the number of non-overlapping superpixels increases the size of the superpixels must decrease.
# The ue increases most if large superpixels overlap only slightly with ground truth segment.
# Reducing the superpixel sizes decrease the chance of that happening.

import os
import math
import numpy as np
from skimage import io, transform, feature
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.segmentation import slic

def main():
    rgb_name = "0001_rgb.png"
    label_name = "0001_label.png"

    rgb_image = io.imread(os.path.join(os.path.dirname(__file__), rgb_name))
    label_image = io.imread(os.path.join(os.path.dirname(__file__), label_name))
    options = [
        (4,10),
        (8,10),
        (16,10),
        (32,10),
        (64,10),
        (4,10),
        (8,20),
        (16,30),
        (32,40),
        (64,50),
    ]
    segmentations = [slic(rgb_image, n_segments=n, compactness=c) for(n,c) in options]

    undersegmentation_errors = [calc_undersegmentation(label_image, segmentation) for segmentation in segmentations]
    ue_averages = [np.average(ues) for (_,ues) in undersegmentation_errors]
    number_of_segments = np.count_nonzero(np.unique(label_image))

    fig, axs = plt.subplots(3,4, figsize=(14,8))
    axs[0,0].imshow(rgb_image)
    axs[0,0].set_title("Original Image")
    axs[0,1].imshow(label_image)
    axs[0,1].set_title(f"Ground Truth, {number_of_segments} segments")
    for (i, seg), (n,c), ue, (labels, errors) in zip(enumerate(segmentations,start=2), options, ue_averages, undersegmentation_errors):
        col = i % 4
        row = i // 4
        axs[row,col].imshow(seg)
        axs[row,col].set_title(f"n_seg={n}, comp={c}, avg(ue)={ue:.2f}")
        print("")
        print(f"n_seg={n}, comp={c}, avg(ue)={ue:.2f}:")
        for (e,label) in zip(errors, labels):
            print(f"UE({label}):{e:.2f}")
    for ax in axs.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()

def calc_undersegmentation(truth, segmentation):
    true_labels = np.unique(truth)
    true_labels = true_labels[true_labels!=0]
    ues = np.array([calc_ue_for_label(truth, segmentation, label) for label in true_labels])
    return true_labels, ues

def calc_ue_for_label(truth, segmentation, label):
    true_mask = truth == label
    truth_area = np.count_nonzero(true_mask)
    matching_labels = np.unique(segmentation[true_mask])
    bleeding = np.isin(segmentation, matching_labels)
    bleeding_area = np.count_nonzero(bleeding)

    return (bleeding_area - truth_area) / truth_area

if __name__ == '__main__':
    main()