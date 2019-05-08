from helpers.deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import os
import cv2
import random
from tqdm import tqdm

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
DREAM_NAME = 'galaxy'
NUM_ITERATIONS = 15

X_SIZE = 1920
Y_SIZE = 1080
X_TRIM = 2
Y_TRIM = 1

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

print(f'Total frames: {VIDEO_LENGTH}')
for i in tqdm(range(0, VIDEO_LENGTH + 1)):
	if not os.path.isfile('dream/{}/img_{}.jpg'.format(DREAM_NAME, i + 1)):
		img_result = load_image(filename='dream/{}/img_{}.jpg'.format(DREAM_NAME, i))

		# this impacts how quick the "zoom" is
		img_result = img_result[0 + X_TRIM:Y_SIZE - Y_TRIM, 0 + Y_TRIM:X_SIZE - X_TRIM]
		img_result = cv2.resize(img_result, (X_SIZE, Y_SIZE))

		img_result[:, :, 0] += random.choice([2, 2, 3])  # reds
		img_result[:, :, 1] += random.choice([2, 2, 3])  # greens
		img_result[:, :, 2] += random.choice([2, 3])  # blues

		img_result = np.clip(img_result, 0.0, 255.0)
		img_result = img_result.astype(np.uint8)

		img_result = recursive_optimize(layer_tensor=TENSOR_LAYERS,
		                                image=img_result,
		                                num_iterations=NUM_ITERATIONS,
		                                step_size=1.0,
		                                rescale_factor=0.7,
		                                num_repeats=1,
		                                blend=0.2)

		img_result = np.clip(img_result, 0.0, 255.0)
		img_result = img_result.astype(np.uint8)
		result = PIL.Image.fromarray(img_result, mode='RGB')
		result.save('dream/{}/img_{}.jpg'.format(DREAM_NAME, i + 1))

		created_count += 1
		if created_count > BATCH_SIZE:
			print('Sequence Done! - Check results and continue')
			break