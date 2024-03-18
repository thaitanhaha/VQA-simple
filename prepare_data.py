from keras.layers import TextVectorization 
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import json
import os
import numpy as np

def load_and_proccess_image(image_path):
	im = img_to_array(load_img(image_path))
	return im / 255 - 0.5

def read_images(paths):
	ims = {}
	for image_id, image_path in paths.items():
		ims[image_id] = load_and_proccess_image(image_path)
	return ims

def extract_paths(dir):
	paths = {}
	for filename in os.listdir(dir):
		if filename.endswith('.png'):
			image_id = int(filename[:-4])
			paths[image_id] = os.path.join(dir, filename)
	return paths

def read_questions(path):
	with open(path, 'r') as file:
		qs = json.load(file)
	texts = [q[0] for q in qs]
	answers = [q[1] for q in qs]
	image_ids = [q[2] for q in qs]
	return (texts, answers, image_ids)

def setup():
	train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')
	test_qs, test_answers, test_image_ids = read_questions('data/test/questions.json')

	with open('data/answers.txt', 'r') as file:
		all_answers = [a.strip() for a in file]

	num_answers = len(all_answers)
	print(f'Found {num_answers} total answers:')
	print(all_answers)

	train_ims = read_images(extract_paths('data/train/images'))
	test_ims  = read_images(extract_paths('data/test/images'))
    
	im_shape = train_ims[0].shape

	vocab_size = 27
	vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='count')
	vectorizer.adapt(train_qs)

	train_X_seqs = vectorizer(train_qs)
	test_X_seqs = vectorizer(test_qs)


	train_X_ims = np.array([train_ims[id] for id in train_image_ids])
	test_X_ims = np.array([test_ims[id] for id in test_image_ids])


	train_answer_indices = [all_answers.index(a) for a in train_answers]
	test_answer_indices = [all_answers.index(a) for a in test_answers]
	train_Y = to_categorical(train_answer_indices)
	test_Y = to_categorical(test_answer_indices)

	return (train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs,
			test_Y, im_shape, vocab_size, num_answers,
			all_answers, test_qs, test_answer_indices, test_image_ids)