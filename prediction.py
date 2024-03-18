
import numpy as np
from model import build_model
from prepare_data import setup


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
