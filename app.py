
import numpy as np
from model import build_model
from prepare_data import setup


train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, all_answers, test_qs, test_answer_indices = setup()

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers)

model.load_weights('model.keras')
predictions = model.predict([test_X_ims, test_X_seqs])


# for idx in range(num_answers):
# 	pred_values = predictions[:, idx]
# 	answer = all_answers[idx]
# 	print(f'\nStatistics for answer {idx}, {answer}')
# 	min = np.amin(pred_values)
# 	max = np.amax(pred_values)
# 	mean = np.mean(pred_values)
# 	print(f'Min: {min}, Max: {max}, Mean: {mean}')

# shapes = []
# yesno = []
# for i in range(num_answers):
#   if (all_answers[i] == 'rectangle' or all_answers[i] == 'circle' or all_answers[i] == 'triangle'):
#     shapes.append(i)
#   elif all_answers[i] == 'yes' or all_answers[i] == 'no':
#     yesno.append(i)

# def return_class(answer):
#   if answer in shapes:
#     return 0
#   if answer in yesno:
#     return 1
#   return 2
# error_matrix = [[0 for _ in range(3)] for _ in range(3)]
# total_errors = 0

# color_error_matrix = [[0 for _ in range(num_answers)] for _ in range(num_answers)]
questions_wrong = {}

for idx in range(len(test_answer_indices)):
  answer = test_answer_indices[idx]
  pred = np.argmax(predictions[idx])
  print(str(test_qs[idx]) + "     " + str(answer) + "     " + str(pred))
#   if not answer == pred:
    # total_errors += 1
    # error_matrix[return_class(answer)][return_class(pred)] += 1
    # color_error_matrix[answer][pred] += 1
    # if (return_class(answer) == 1 and return_class(pred) == 1):
    #   if test_qs[idx] in questions_wrong:
    #     questions_wrong[test_qs[idx]] += 1
    #   else:
    #     questions_wrong[test_qs[idx]] = 1

# print('total error: {}'.format(total_errors / len(test_answer_indices)))
# print('Indexes are, in order, shapes, yes/no, colors')
# print('Rows are class of answer, columns are class of prediction')
# for i in range(3):
#   print('{}\t{}\t{}\n'.format(error_matrix[i][0] / total_errors, error_matrix[i][1] / total_errors, error_matrix[i][2]/ total_errors))
# print('-------------')
# for i in range(num_answers):
#   to_print = ''
#   for j in range(num_answers):
#     to_print += str(color_error_matrix[i][j]) + '\t'
#   print(to_print)
# print('-------------')
# print(questions_wrong)
