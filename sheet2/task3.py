# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz
#
# What advantage does using this approach based on image pyramids have over the integral
# image method from assignment sheet 1?
# When applying a filter to a signal the computational complexity depends both on the 
# filter size and on the signal size.
# So instead of applying multiple increasing filter sizes to a single large image
# it is more efficient to apply a single smaller filter to multiple smaller images.
# For the result only the relative size between the image and the filter matters.
# The gaussian filter is a lowpass filter so we know we can discard high frequencies anyway
# by downsampling the image.

import os
import numpy as np
from skimage import io, transform
from skimage.util import random_noise
from skimage.color import rgba2rgb, rgb2gray
from skimage.filters import gaussian, sobel
from skimage.transform import pyramid_gaussian, resize

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
	print(os.path.basename(__file__))
	print("".join(open(__file__, 'r').readlines()[0:13]))
	image_name = 'visual_attention.png'
	image = io.imread(os.path.join(os.path.dirname(__file__), image_name))
	image_gray = rgb2gray(rgba2rgb(image))
	w,h = image_gray.shape
	image_pyramid_center = list(pyramid_gaussian(image_gray, max_layer=4, sigma=9))
	image_pyramid_surround = list(pyramid_gaussian(image_gray, max_layer=4, sigma=16))

	on_off_contrast = [np.clip(c-s,0,1) for c,s in zip(image_pyramid_center, image_pyramid_surround)]
	off_on_contrast = [np.clip(s-c,0,1) for c,s in zip(image_pyramid_center, image_pyramid_surround)]

	on_off_contrast_upsampled = np.array([resize(l, image_gray.shape) for l in on_off_contrast])
	off_on_contrast_upsampled = np.array([resize(l, image_gray.shape) for l in off_on_contrast])

	on_off_avg = on_off_contrast_upsampled.mean(0)
	off_on_avg = off_on_contrast_upsampled.mean(0)

	all_layers = np.stack((on_off_avg, off_on_avg))
	all_layers_average = all_layers.mean(0)

	fig, ax = plt.subplots(5, 4, figsize=(10,7))
	fig.suptitle("Sheet 2, Task 3: Image Pyramids")

	ax[0, 0].imshow(image_gray, cmap='gray')
	ax[0, 0].set_title("Original Grayscale")
	for j in range(4):
		ax[1, j].imshow(image_pyramid_center[j], cmap='gray')
		ax[1, j].set_title(f"Center Pyramid j={j}")
		ax[2, j].imshow(image_pyramid_surround[j], cmap='gray')
		ax[2, j].set_title(f"Surround Pyramid j={j}")
		ax[3, j].imshow(on_off_contrast[j], cmap='gray')
		ax[3, j].set_title(f"on/off contr. j={j}")
		ax[4, j].imshow(off_on_contrast[j], cmap='gray')
		ax[4, j].set_title(f"off/on contr. j={j}")

	ax[0, 1].imshow(on_off_avg, cmap='gray')
	ax[0, 1].set_title(f"on/off avg.")
	ax[0, 2].imshow(off_on_avg, cmap='gray')
	ax[0, 2].set_title(f"off/on avg.")
	ax[0, 3].imshow(all_layers_average, cmap='gray')
	ax[0, 3].set_title(f"conspicuity")
	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

