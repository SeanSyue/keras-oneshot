import os
import pickle
from SiameseModel.Loader import SiameseLoader
from SiameseModel.Network import SiameseNet

BATCH_SIZE = 32
ITER = 100000
N_WAY = 20  # how many classes for testing one-shot tasks
N_VAL = 250  # how many one-shot tasks to validate on
EVAL_EVERY = 50  # interval for evaluating on one-shot tasks
SAVE_EVERY = 2000

PICKLE_PATH = os.path.join('data')
TRAIN_FILE = os.path.join(PICKLE_PATH, 'train.pickle')
VAL_FILE = os.path.join(PICKLE_PATH, 'val.pickle')

EXPORT_WEIGHTS_PATH = 'weights'
IMPORT_WEIGHTS = 'weights/siamese_weights_98000.h5'  # Remain 'None' if train from scratch


def main():
    if not os.path.isdir(EXPORT_WEIGHTS_PATH):
        os.makedirs(EXPORT_WEIGHTS_PATH)
    # LOAD WEIGHTS ================================================================================================
    with open(TRAIN_FILE, 'rb') as f:
        (X, c) = pickle.load(f)

    with open(VAL_FILE, 'rb') as f:
        (Xval, cval) = pickle.load(f)

    print("training alphabets:\n"
          "{}\n"
          "validation alphabets:\n"
          "{}".format(c.keys, cval.keys()))

    # TRAIN LOOP ==================================================================================================
    print("training")
    loader = SiameseLoader(PICKLE_PATH, BATCH_SIZE)
    siamese_net = SiameseNet('weights/siamese_weights_98000.h5').setup()
    for i in range(1, ITER):
        (inputs, targets) = loader.get_batch(BATCH_SIZE)
        loss = siamese_net.train_on_batch(inputs, targets)
        print(" -- Iter: {} -- Loss: {} -- ".format(i, loss))
        if i % EVAL_EVERY == 0:
            loader.test_oneshot(siamese_net, N_WAY, N_VAL, verbose=True)
        if i % SAVE_EVERY == 0:
            siamese_net.save_weights(os.path.join(EXPORT_WEIGHTS_PATH, 'siamese_weights_{}.h5'.format(i)))
            print(" ==== Model saved on {} ====".format(i))


if __name__ == '__main__':
    main()
