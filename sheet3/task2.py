# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import os
import numpy as np
from sklearn.linear_model import RANSACRegressor
from skimage import io, transform, feature, filters
from skimage.util import random_noise
from skimage.filters import gaussian, sobel
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
	image_name = 'yellow_horizon.jpg'
	image_orig = io.imread(os.path.join(os.path.dirname(__file__), image_name))

	angles = [0, -12, 17]
	fig, ax = plt.subplots(len(angles),4, figsize=(14,9))
	fig.suptitle("Sheet 3, Task 2: RANSAC")

	for i, a in enumerate(angles):
		image_rot = transform.rotate(image_orig, a, mode='reflect')
		image_gray = rgb2gray(image_rot)
		image_smooth = filters.gaussian(image_gray, sigma=3)
		image_sobel = filters.sobel_h(image_smooth)


		ax[i, 0].imshow(image_rot)
		ax[i, 0].set_title(f"Original (rotation={a}°)")
		ax[i, 1].imshow(image_gray, cmap='gray')
		ax[i, 1].set_title(f"Gray and Horizon (rotation={a}°)")
		ax[i, 2].imshow(image_smooth, cmap='gray')
		ax[i, 2].set_title(f"Smoothed(sig=3) (rotation={a}°)")
		ax[i, 3].imshow(image_sobel, cmap='gray')
		ax[i, 3].set_title(f"Sobel and candidates (rotation={a}°)")

		candidates, l1, l2 = predict_horizon(image_sobel)

		for x, y in candidates:
			ax[i, 3].axvline(x, color='r', lw=0.5)
			ax[i, 3].plot(x, y, 'sr')

		ax[i, 1].plot(l1, l2, '-y', lw=2)

	fig.tight_layout()
	plt.show()

def predict_horizon(image_sobel):
	height, width = image_sobel.shape

	hop_size = width // 20
	minimums = image_sobel[:,::hop_size].argmin(0)
	candidates = [(i*hop_size, y) for i, y in enumerate(minimums)]

	ransac = RANSACRegressor(min_samples=2)
	ransac.fit(np.arange(0, width, hop_size).reshape(-1, 1), minimums)
	score = ransac.score(np.arange(0, width, hop_size).reshape(-1, 1), minimums)

	return candidates, (0,width-1), (ransac.predict([[0]])[0], ransac.predict([[width-1]])[0])

if __name__ == '__main__':
	main()

