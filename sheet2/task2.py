# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz
#
# Why is edge detection improved by applying smoothing?
# Answer: Edge detection is improvied by applying smoothing because
# high frequency noise in a signal leads to the signal values jumping up and down
# so it creates many position and negative gradients in the signal.
# Gradients are also what we interepret is edges. So when detecting edges we try to find
# gradients but in a noisy signal we get many false-positive results. 

import os
import numpy as np
from skimage import io, transform
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
	print(os.path.basename(__file__))
	print("".join(open(__file__, 'r').readlines()[0:10]))
	image_name = 'woman.png'
	sigma = 1
	var = 0.01
	original_image = io.imread(os.path.join(os.path.dirname(__file__), image_name))
	noisy_image = random_noise(original_image, var=var)
	filtered_image = gaussian(noisy_image, sigma=sigma)
	original_sobel = sobel(original_image)
	noisy_sobel = sobel(noisy_image)
	filtered_sobel = sobel(filtered_image)

	noisy_hist = np.histogram(np.reshape(noisy_sobel, (-1,1)),bins=60)
	filtered_hist = np.histogram(np.reshape(filtered_sobel, (-1,1)),bins=60)

	noisy_threshold = 0.13 # noisy_hist[1][noisy_hist[0].argmax()]
	filtered_threshold = 0.058 # filtered_hist[1][filtered_hist[0].argmax()]

	fig, ax = plt.subplots(4, 3, figsize=(10,7))
	fig.suptitle("Sheet 2, Task 2: Edge Detection")
	ax[0, 0].imshow(original_image, cmap='gray')
	ax[0, 0].set_title(f"Original Gray")
	ax[0, 1].imshow(noisy_image, cmap='gray')
	ax[0, 1].set_title(f"Noisy(var={var})")
	ax[0, 2].imshow(filtered_image, cmap='gray')
	ax[0, 2].set_title(f"Gauss-filtered (sigma={sigma})")
	ax[1, 0].imshow(original_sobel, cmap='gray')
	ax[1, 0].set_title(f"Original + sobel")
	ax[1, 1].imshow(noisy_sobel, cmap='gray')
	ax[1, 1].set_title(f"Noisy + sobel")
	ax[1, 2].imshow(filtered_sobel, cmap='gray')
	ax[1, 2].set_title(f"Gauss + sobel")
	ax[2, 0].hist(np.reshape(original_sobel, (-1,1)),bins=60)
	ax[2, 0].set_title(f"Original Histogram (60bins)")
	ax[2, 1].hist(np.reshape(noisy_sobel, (-1,1)),bins=60)
	ax[2, 1].set_title(f"Noisy Histogram (60bins)")
	ax[2, 2].hist(np.reshape(filtered_sobel, (-1,1)),bins=60)
	ax[2, 2].set_title(f"Gauss Histogram (60bins)")
	ax[3, 0].set_visible(False)
	ax[3, 1].imshow(noisy_sobel>noisy_threshold, cmap='gray')
	ax[3, 1].set_title(f"noisy_sobel>noisy_threshold={noisy_threshold}")
	ax[3, 2].imshow(filtered_sobel>filtered_threshold, cmap='gray')
	ax[3, 2].set_title(f"filtered_sobel>filtered_threshold={filtered_threshold}")
	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

