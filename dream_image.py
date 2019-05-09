from helpers.deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import PIL.ImageStat
import os
import random
from tqdm import tqdm
import math

"""
layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
layer 6: dogs, bears, cute animals.
layer 7: faces, buildings
layer 8: fish begin to appear, frogs/reptilian eyes.
layer 10: Monkies, lizards, snakes, duck
"""
LAYER_NUMBER = 3
NUM_ITERATIONS = 17

X_SIZE = 1920
Y_SIZE = 1080
X_TRIM = 2
Y_TRIM = 1

BRIGHT_MIN = 70
BRIGHT_MAX = 90

FRAMES_PER_SEC = 30

# Constants / dont change
SEC = 1 * FRAMES_PER_SEC
MINUTE = 60 * SEC
HOUR = 60 * MINUTE
#########################

VIDEO_LENGTH = 3 * MINUTE
BATCH_SIZE = 50

TENSOR_LAYERS = model.layer_tensors[LAYER_NUMBER]
created_count = 0

def brightness(im_file):
	im = PIL.Image.open(im_file)
	stat = PIL.ImageStat.Stat(im)
	r, g, b = stat.mean
	return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

print(f'Total frames: {VIDEO_LENGTH}')
for i in tqdm(range(0, VIDEO_LENGTH + 1)):
	if not os.path.isfile('dream/img_{}.jpg'.format(i + 1)):
		img_result = load_image(filename='dream/img_{}.jpg'.format(i))

		if random.randint(0, 500) == 1:
			choice = random.choice([1, 2, 3, 4, 6, 7, 8, 10])
			TENSOR_LAYERS = model.layer_tensors[choice]
			if choice < 4:
				NUM_ITERATIONS = 17
			elif choice < 7:
				NUM_ITERATIONS = 22
			elif choice == 8:
				NUM_ITERATIONS = 26
			else:
				NUM_ITERATIONS = 30

		# this impacts how quick the "zoom" is
		img_result = img_result[0 + X_TRIM:Y_SIZE - Y_TRIM, 0 + Y_TRIM:X_SIZE - X_TRIM]
		br = brightness('dream/img_{}.jpg'.format(i))

		if BRIGHT_MAX < br < BRIGHT_MIN:
			# Image is too dark
			img_result[:, :, 0] += random.choice([2, 3, 4])  # reds
			img_result[:, :, 1] += random.choice([3, 4])  # greens
			img_result[:, :, 2] += random.choice([3, 4])  # blues
		else:
			# Image is too bright
			img_result[:, :, 0] += random.choice([2, 3])  # reds
			img_result[:, :, 1] += random.choice([2, 3])  # greens
			img_result[:, :, 2] += random.choice([2, 3])  # blues

		img_result = np.clip(img_result, 0.0, 255.0)
		img_result = img_result.astype(np.uint8)

		img_result = recursive_optimize(layer_tensor=TENSOR_LAYERS,
		                                image=img_result,
		                                num_iterations=NUM_ITERATIONS,
		                                step_size=1.0,
		                                rescale_factor=0.75,
		                                num_repeats=1,
		                                blend=0.2)

		img_result = np.clip(img_result, 0.0, 255.0)
		img_result = img_result.astype(np.uint8)
		result = PIL.Image.fromarray(img_result, mode='RGB')
		result.save('dream/img_{}.jpg'.format(i + 1))

		created_count += 1
		if created_count > BATCH_SIZE:
			print('Sequence Done! - Check results and continue')
			break
