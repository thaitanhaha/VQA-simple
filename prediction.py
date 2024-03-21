from keras.layers import TextVectorization 
import numpy as np
import os, sys
from model import build_model
from prepare_data import setup, load_and_proccess_image, read_questions, proccess_image


def test_with_generated_data():
  train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, all_answers, test_qs, test_answer_indices, test_image_ids = setup()
  print('\n--- Building model...')
  model = build_model(im_shape, vocab_size, num_answers)

  model.load_weights('model.keras')
  predictions = model.predict([test_X_ims, test_X_seqs])

  wrong_count = 0
  for idx in range(len(test_answer_indices)):
    answer = all_answers[test_answer_indices[idx]]
    pred = all_answers[np.argmax(predictions[idx])]
    if answer != pred:
      print(f"Image: {test_image_ids[idx]:<5} {test_qs[idx]:<45} Answer: {answer:<15} Prediction: {pred}")
      wrong_count += 1
  print('----------------------')
  print("Total prediction: " + str(len(test_answer_indices)))
  print("Wrong prediction: " + str(wrong_count))

def test_with_custom_data(file, question):
  # print(load_and_proccess_image('custom_data/0.png'))
  print('\n--- Building model...')
  model = build_model((64, 64, 3), 27, 13)

  model.load_weights('model.keras')

  train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')

  vectorizer = TextVectorization(max_tokens=27, output_mode='count')
  vectorizer.adapt(train_qs)
  test_X_seqs = vectorizer([question])

  if type(file) is str:
    if os.path.isfile(file) and file.endswith('.png'):
      ims = load_and_proccess_image(file)
  else:
    ims = proccess_image(file) 
  test_X_ims = np.array([ims])

  predictions = model.predict([test_X_ims, test_X_seqs])
  all_answers = ['teal', 'black', 'rectangle', 'green', 'triangle', 'circle', 'blue', 'gray', 'red', 'yes', 'no', 'yellow', 'brown']
  print(all_answers[np.argmax(predictions[0])])
  return all_answers[np.argmax(predictions[0])]

def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        test_with_generated_data()
    elif len(sys.argv) == 4 and sys.argv[1] == 'custom':
        test_with_custom_data(sys.argv[2], sys.argv[3])
    else:
        print("Invalid usage. Please use one of the following formats:")
        print("python run.py test")
        print("python run.py custom filepath question")

if __name__ == "__main__":
    main()