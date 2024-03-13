from keras.callbacks import ModelCheckpoint
from model import build_model
from prepare_data import setup


# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup()

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers)
# checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
checkpoint = ModelCheckpoint('model.keras', save_best_only=True)

print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=4,
  callbacks=[checkpoint],
)
