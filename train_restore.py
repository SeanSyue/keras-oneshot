import os
import pickle
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import  Adam
from SiameseModel.Loader import SiameseLoader


input_shape = (105, 105, 1)
# build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
convnet.add(Flatten())
convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3)))

left_input = Input(input_shape)
right_input = Input(input_shape)
# call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
# layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
# call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(0.00006))

siamese_net.load_weights('weights/siamese_weights_98000.h5')


# ======================================================================================================================
pickle_path = os.path.join('data')

with open(os.path.join(pickle_path, "train.pickle"), "rb") as f:
    (X, c) = pickle.load(f)

with open(os.path.join(pickle_path, "val.pickle"), "rb") as f:
    (Xval, cval) = pickle.load(f)

print("training alphabets:\n"
      "{}\n"
      "validation alphabets:\n"
      "{}".format(c.keys, cval.keys()))


# ======================================================================================================================
# Instantiate the class
batch_size = 32
loader = SiameseLoader(pickle_path, batch_size)

# Training loop
n_iter = 900001
evaluate_every = 50  # interval for evaluating on one-shot tasks

N_way = 20  # how many classes for testing one-shot tasks>
n_val = 250  # how mahy one-shot tasks to validate on?
save_every = 2000
weights_path = 'weights'
if not os.path.isdir(weights_path):
    os.makedirs(weights_path)
print("training")
for i in range(100001, n_iter):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = siamese_net.train_on_batch(inputs, targets)
    print(" -- Iter: {} -- Loss: {} -- ".format(i, loss))
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(siamese_net, N_way, n_val, verbose=True)
    if i % save_every == 0:
        siamese_net.save_weights(os.path.join(weights_path, 'siamese_weights_{}.h5'.format(i)))
        print(" ==== Model saved on {} ====".format(i))
