# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import os
import numpy as np
from skimage import io, transform
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
	image_name = 'woman.png'
	original_image = io.imread(os.path.join(os.path.dirname(__file__), image_name))
	noisy_image = random_noise(original_image, var=0.01)
	filtered_image = gaussian(noisy_image, sigma=1)
	original_sobel = sobel(original_image)
	noisy_sobel = sobel(noisy_image)
	filtered_sobel = sobel(filtered_image)

	noisy_hist = np.histogram(np.reshape(noisy_sobel, (-1,1)),bins=60)
	filtered_hist = np.histogram(np.reshape(filtered_sobel, (-1,1)),bins=60)

	noisy_threshold = 0.13 # noisy_hist[1][noisy_hist[0].argmax()]
	filtered_threshold = 0.058 # filtered_hist[1][filtered_hist[0].argmax()]

	print(noisy_threshold)
	print(filtered_threshold)

	fig, ax = plt.subplots(4, 3)
	fig.suptitle("Sheet 2, Task 2: Edge Detection")
	ax[0, 0].imshow(original_image, cmap='gray')
	ax[0, 1].imshow(noisy_image, cmap='gray')
	ax[0, 2].imshow(filtered_image, cmap='gray')
	ax[1, 0].imshow(original_sobel, cmap='gray')
	ax[1, 1].imshow(noisy_sobel, cmap='gray')
	ax[1, 2].imshow(filtered_sobel, cmap='gray')
	ax[2, 0].hist(np.reshape(original_sobel, (-1,1)),bins=60)
	ax[2, 1].hist(np.reshape(noisy_sobel, (-1,1)),bins=60)
	ax[2, 2].hist(np.reshape(filtered_sobel, (-1,1)),bins=60)
	ax[3, 0].set_visible(False)
	ax[3, 1].imshow(noisy_sobel>noisy_threshold, cmap='gray')
	ax[3, 2].imshow(filtered_sobel>filtered_threshold, cmap='gray')
	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

