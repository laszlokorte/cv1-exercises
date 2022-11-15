# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import os
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import match_template

import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
	print(os.path.basename(__file__))
	print("".join(open(__file__, 'r').readlines()[0:3]))

	template_name = 'template.jpg'
	panorame_name = 'panorama.jpg'
	template = io.imread(os.path.join(os.path.dirname(__file__), template_name))
	panorama = io.imread(os.path.join(os.path.dirname(__file__), panorame_name))


	template_gray = rgb2gray(template)
	panorama_gray = rgb2gray(panorama)
	template_gray_flipped = np.flip(template_gray,axis=1)

	tpl_height, tpl_width = template_gray.shape

	result_a = match_template(panorama_gray, template_gray)
	result_b = match_template(panorama_gray, template_gray_flipped)
	xa, ya = np.unravel_index(np.argmax(result_a), result_a.shape)[::-1]
	xb, yb = np.unravel_index(np.argmax(result_b), result_b.shape)[::-1]
	
	fig, ax = plt.subplots(2, 2, figsize=(10,7))
	fig.suptitle("Sheet 2, Task 1: Template matching")

	ax[0, 0].imshow(template_gray, cmap='gray')
	ax[0, 0].set_title(f"Template Image")
	ax[0, 1].imshow(panorama_gray, cmap='gray')
	ax[0, 1].set_title(f"Panorama Image with maximal correlation")
	rect = plt.Rectangle((xa, ya), tpl_width, tpl_height, edgecolor='r', facecolor='none')
	ax[0, 1].add_patch(rect)

	ax[1, 0].imshow(template_gray_flipped, cmap='gray')
	ax[1, 0].set_title(f"Flipped Template")
	ax[1, 1].imshow(panorama_gray, cmap='gray')
	ax[1, 1].set_title(f"Panorama Image with max correlation at wrong position")
	rect = plt.Rectangle((xb, yb), tpl_width, tpl_height, edgecolor='r', facecolor='none')
	ax[1, 1].add_patch(rect)

	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

