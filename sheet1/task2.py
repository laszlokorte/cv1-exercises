import os
from skimage import io, transform
from skimage.color import rgba2rgb, rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
	window_sizes = [
		(3, 7),
		(11, 21),
		(31, 51),
	]
	file_name = 'visual_attention_ds.png'

	flower = io.imread(os.path.join(os.path.dirname(__file__), file_name))
	flower_gray = rgb2gray(rgba2rgb(flower))
	flower_integral = transform.integral.integral_image(flower_gray)

	height, width = flower_gray.shape

	rows, columns = np.indices((height, width))

	def is_inside(inner, outer):
		return (outer[0] <= inner[0]) & (inner[0] < outer[1]) & (outer[0] <= inner[1]) & (inner[1] < outer[1])

	fig, ax = plt.subplots(2, len(window_sizes))

	fig.suptitle("Sheet 1, Task 2: Surround-Center Contract")
	ax[0, 0].imshow(flower)
	ax[0, 0].set_title("Original RGB")
	ax[0, 1].imshow(flower_gray, cmap='gray')
	ax[0, 1].set_title("Grayscale")
	for x in range(2, len(window_sizes)):
		ax[0, x].set_visible(False)

	print("Calculating Center/Surround Contrast")

	for i, (inner, outer) in enumerate(tqdm(window_sizes)):
		result = np.zeros_like(flower_gray)
		inner_half = int((inner - 1)/2)
		outer_half = int((outer - 1)/2)

		r0c, c0c, r1c, c1c = rows - inner_half, columns - inner_half, rows + inner_half, columns + inner_half
		r0o, c0o, r1o, c1o = rows - outer_half, columns - outer_half, rows + outer_half, columns + outer_half

		cutout = is_inside((c0o, c1o), (0, width)) & is_inside((r0o, r1o), (0, height))

		center_sum = transform.integral.integrate(flower_integral, np.array(list(zip(r0c[cutout], c0c[cutout]))), np.array(list(zip(r1c[cutout], c1c[cutout]))))
		surround_sum = transform.integral.integrate(flower_integral, np.array(list(zip(r0o[cutout], c0o[cutout]))), np.array(list(zip(r1o[cutout], c1o[cutout]))))
		diff = surround_sum - center_sum
		result[cutout] = diff

		ax[1, i].set_title(f"Suround/Center: {outer}/{inner}")
		ax[1, i].imshow(result, cmap='gray')

	fig.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

