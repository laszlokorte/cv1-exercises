from skimage import io
from skimage import transform
from skimage.color import rgba2rgb, rgb2gray
import matplotlib.pyplot as plt
import numpy as np

flower = io.imread('visual_attention_ds.png')
flower_gray = rgb2gray(rgba2rgb(flower))
flower_integral = transform.integral.integral_image(flower_gray)

height, width = flower_gray.shape

window_sizes = [
	(3, 7),
	(11, 21),
	(31, 51),
]

def is_inside(inner, outer):
	return outer[0] <= inner[0] < outer[1] and outer[0] <= inner[1] < outer[1]

fig, ax = plt.subplots(2, 3)

ax[0, 0].imshow(flower)
ax[0, 0].set_title("Original RGB")
ax[0, 1].imshow(flower_gray, cmap='gray')
ax[0, 1].set_title("Grayscale")
ax[0, 2].set_visible(False)

for i, (inner, outer) in enumerate(window_sizes):
	result = np.zeros_like(flower_gray)
	inner_half = int((inner - 1)/2)
	outer_half = int((outer - 1)/2)
	for row in range(height):
		for col in range(width):
			r0c, c0c, r1c, c1c = row - inner_half, col - inner_half, row + inner_half, col + inner_half
			r0o, c0o, r1o, c1o = row - outer_half, col - outer_half, row + outer_half, col + outer_half
			diff = 0
			if is_inside((c0o, c1o), (0, width)) and is_inside((r0o, r1o), (0, height)):
					center_sum = transform.integral.integrate(flower_integral, (r0c, c0c), (r1c, c1c))
					surround_sum = transform.integral.integrate(flower_integral, (r0o, c0o), (r1o, c1o))
					diff = surround_sum - center_sum
			result[row,col] = diff

	ax[1, i].set_title(f"Suround/Center: {outer}/{inner}")
	ax[1, i].imshow(result, cmap='gray')

fig.tight_layout()
plt.show()
