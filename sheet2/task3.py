import os
import numpy as np
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import pyramid_gaussian, resize

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
	image_name = 'visual_attention.png'
	image = io.imread(os.path.join(os.path.dirname(__file__), image_name))
	image_gray = rgb2gray(rgba2rgb(image))
	w, h = image_gray.shape
	
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

	# alternatively skip calculating the on_off and off_on contrast separately:
	# mean[clip(a-b), clip(b-a)] =  abs(b-a)/2
	abs_contrast = [np.abs(c-s)/2 for c,s in zip(image_pyramid_center, image_pyramid_surround)]
	abs_contrast_upsampled = np.array([resize(l, image_gray.shape) for l in abs_contrast])
	abs_avg = abs_contrast_upsampled.mean(0)

	plot_all(image, image_pyramid_center, image_pyramid_surround, on_off_contrast, off_on_contrast, on_off_avg, off_on_avg, all_layers_average, abs_avg)


def plot_all(image, image_pyramid_center, image_pyramid_surround, on_off_contrast, off_on_contrast, on_off_avg, off_on_avg, all_layers_average, abs_avg):
	fig, ax = plt.subplots(5, 5, figsize=(16,12))
	fig.suptitle("Sheet 2, Task 3: Image Pyramids")
	
	hide_all_axis(ax)
	add_extra_labels(ax)

	ax[0, 1].imshow(image)
	ax[0, 1].set_title("Original Color")
	ax[0, 1].set_visible(True)

	plot_image_gray(ax[0, 0], image_pyramid_center[0], "Original Grayscale")

	for a, j in enumerate(range(1,5)):
		plot_image_gray(ax[1, a], image_pyramid_center[j], f'Center Pyramid j={j}')
		plot_image_gray(ax[2, a], image_pyramid_surround[j], f'Surround Pyramid j={j}')
		plot_image_gray(ax[3, a], on_off_contrast[j], f'on/off contr. j={j}')
		plot_image_gray(ax[4, a], off_on_contrast[j], f'off/on contr. j={j}')

	plot_image_gray(ax[3, 4], on_off_avg, f'on/off avg.')
	plot_image_gray(ax[4, 4], off_on_avg, f'off/on avg.')
	plot_image_gray(ax[0, 4], all_layers_average, f'conspicuity')
	plot_image_gray(ax[0, 3], abs_avg, f'conspicuity(abs)')

	for a in range(1, 5):
		ax[a, 0].xaxis.set_major_locator(ticker.NullLocator())
		ax[a, 0].yaxis.set_major_locator(ticker.NullLocator())

	fig.tight_layout()
	plt.show()


def plot_image_gray(ax, image, label):
	ax.xaxis.set_major_locator(ticker.NullLocator())
	ax.yaxis.set_major_locator(ticker.NullLocator())
	ax.imshow(image, cmap='gray', interpolation='none')
	ax.set_title(f"{label}, {format_shape(image.shape)}")
	ax.set_visible(True)


def format_shape(s):
	return "x".join(map(str, s))


def hide_all_axis(ax):
	for a in ax.reshape(-1):
		a.set_visible(False)


def add_extra_labels(ax):
	ax[1, 0].set_ylabel("C", rotation=0, fontsize=15, labelpad=40,loc='center')
	ax[2, 0].set_ylabel("S", rotation=0, fontsize=15, labelpad=40,loc='center')
	ax[3, 0].set_ylabel("clip(C-S)", rotation=0, fontsize=15, labelpad=40,loc='center')
	ax[4, 0].set_ylabel("clip(S-C)", rotation=0, fontsize=15, labelpad=40,loc='center')

if __name__ == '__main__':
	main()