# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import os
import numpy as np
from skimage import io, transform, feature
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
	image_name = 'coins.jpg'
	image_gray = io.imread(os.path.join(os.path.dirname(__file__), image_name), as_gray=True)
	
	image_res_mm_per_px = 0.12
	coins_in_mm = [
		('10marks', 27.25),
		('5marks', 24.25),
		('1mark', 22.25),
		('50penni', 19.70),
		('10penni', 16.30),
	]
	coins_in_px = [(name, int(size/image_res_mm_per_px)) for (name, size) in coins_in_mm]

	image_edges = feature.canny(image_gray)
	hough_spaces = np.concatenate([transform.hough_circle(image_edges, size/2) for (name, size) in coins_in_px])
	radii = [size/2 for (name, size) in coins_in_px]
	peaks = transform.hough_circle_peaks(hough_spaces, radii, num_peaks=2, total_num_peaks=10, normalize=True)


	fig, ax = plt.subplots(len(coins_in_px), 2, figsize=(8,10))
	fig.suptitle("Sheet 3, Task 1: Hough transform")
	
	ax[0, 0].imshow(image_gray, cmap='gray')
	ax[0, 0].set_title(f"Coins Gray & Results")
	ax[1, 0].imshow(image_edges, cmap='gray')
	ax[1, 0].set_title(f"Coins Edges")
	ax[2, 0].set_visible(False)
	ax[3, 0].set_visible(False)
	ax[4, 0].set_visible(False)

	for i, hough in enumerate(hough_spaces):
		ax[i, 1].imshow(hough, cmap='gray')
		ax[i, 1].set_title(f"Hough Space ({coins_in_px[i][0]}, r={coins_in_px[i][1]/2}px)")

	accum, cx, cy, rad = peaks
	for (accum, cx, cy, rad) in zip(accum, cx, cy, rad):
		circle = plt.Circle((cx, cy), rad, edgecolor='r', facecolor='none')
		ax[0, 0].add_patch(circle)

	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

