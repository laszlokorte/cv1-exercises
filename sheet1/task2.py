from skimage import io
from skimage import transform
from skimage.color import rgba2rgb, rgb2gray
import matplotlib.pyplot as plt
import numpy as np

flower = io.imread('visual_attention_ds.png')
flower_gray = rgb2gray(rgba2rgb(flower))
flower_integral = transform.integral.integral_image(flower_gray)

print(flower_gray)

height, width = flower_gray.shape

result = np.zeros_like(flower_gray)

plt.imshow(flower_gray)
plt.show()

window_sizes = [
	(11, 21),
	(3, 7),
	(31, 51),
]

def is_inside(inner, outer):
	return outer[0] <= inner[0] < outer[1] and outer[0] <= inner[1] < outer[1]

for inner, outer in window_sizes:
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

	plt.imshow(result)
	plt.show()

