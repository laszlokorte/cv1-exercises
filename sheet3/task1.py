# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import os
import math
import numpy as np
from skimage import io, transform, feature
from skimage.util import random_noise
from skimage.filters import gaussian, sobel

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
	# Load Image
	image_name = 'coins.jpg'
	image_gray = io.imread(os.path.join(os.path.dirname(__file__), image_name), as_gray=True)
	
	# Configure expected coin sizes
	image_res_mm_per_px = 0.12
	coins_in_mm = [
		('10marks', 27.25),
		('5marks', 24.25),
		('1mark', 22.25),
		('50penni', 19.70),
		('10penni', 16.30),
	]
	coins_in_px = [(name, int(size/image_res_mm_per_px)) for (name, size) in coins_in_mm]
	radii = [size/2 for (name, size) in coins_in_px]

	# Apply Hough-Transform:
	# 1. Edge Detection
	image_edges = feature.canny(image_gray)
	# 2. Buildd Hough-Space
	hough_spaces = np.concatenate([transform.hough_circle(image_edges, size/2) for (name, size) in coins_in_px])
	# 3. Search for peaks in Hough-Space
	accums, cxs, cys, rads = transform.hough_circle_peaks(hough_spaces, radii, num_peaks=2, total_num_peaks=10, normalize=True)

	# Setup Plots
	number_of_coins = len(coins_in_px)
	columns = math.ceil(number_of_coins/2)
	fig, axs = plt.subplots(2, columns+1,  figsize=(12,5))
	hide_all_axis(axs)
	fig.suptitle("Sheet 3, Task 1: Hough transform")
	
	# Plot original image and edges
	axs[0, 0].imshow(image_gray, cmap='gray')
	axs[0, 0].set_title(f"Coins Gray & Results")
	axs[0, 0].set_visible(True)
	axs[1, 0].imshow(image_edges, cmap='gray')
	axs[1, 0].set_title(f"Coins Edges")
	axs[1, 0].set_visible(True)

	# Plot Hough-Spaces and peaks
	for i, hough in enumerate(hough_spaces):
		ax = axs[i//columns, i%columns + 1]
		ax.imshow(hough, cmap='gray')
		ax.set_title(f"Hough Space ({coins_in_px[i][0]}, r={coins_in_px[i][1]/2}px)")
		ax.set_visible(True)
		peak = plt.Circle((cxs[i], cys[i]), 10, edgecolor='cyan', facecolor='none')
		ax.add_patch(peak)

	# Highlight results in original image
	for (accum, cx, cy, rad) in zip(accums, cxs, cys, rads):
		circle = plt.Circle((cx, cy), rad, edgecolor='r', facecolor='none')
		axs[0, 0].add_patch(circle)

	fig.tight_layout()
	plt.show()

def hide_all_axis(axs):
	for a in axs:
		for x in a:
			x.set_visible(False)

if __name__ == '__main__':
	main()
