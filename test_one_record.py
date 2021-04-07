import sys
import pickle
import DAQ_BG
from CNN_classifier import CNN_Classifier
import numpy as np
from keras.utils import to_categorical
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from preprocessing_and_adding import preprocess_one_record
from DAQ_BG import rec
from distance import correlation_dist
import pickle

def test_spec(spec_imgs, weight_file_name):
    num_filters = 10
    filter_size = 3
    num_classes = (3,)
    input_shape = spec_imgs[0].shape # need to be (36, 1030) after adding zero columns

    CNN_c = CNN_Classifier(input_shape, num_classes, filter_size, num_filters, task='classification', load=True,
                           weight_file=weight_file_name)

    preds_3classes = CNN_c.predict(np.array(spec_imgs))
    preds_2classes = [y[0] > 0.5 for y in preds_3classes]
    return preds_3classes, preds_2classes

if __name__ == '__main__':
    sys.setrecursionlimit(500000)
    # paths
    # d = d'C:\Users\Administrator\PycharmProjects\SonarNano\data'
    # file = '8.mat'
    #weight_file_name = d'\saved_CNN_clasifier_with_zero_columns_learn123_test4_3classes_71_2classes_83.7.h5'
    weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'


    # load data
    counter = 0
    record = rec()
    counter, x_test = preprocess_one_record(record, counter, 'noise')

    # predict
    preds_3classes, preds_2classes = test_spec(x_test, weight_file_name)
    preds_3classes_012 = [np.argmax(y) for y in preds_3classes]
    preds_2classes_01 = 1*(~np.array(preds_2classes))
    print(preds_3classes)
    transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end, DAQ_BG.update_freq,
                                         DAQ_BG.trapRel)
    D = correlation_dist(transmition_Chirp, record)


    print(np.shape(record))
    pickle.dump(transmition_Chirp, open("transmition_Chirp.p", "wb"))
    pickle.dump(record, open("x_test_grape_only.p", "wb"))


    print('record', record)
    plt.plot(record)
    plt.show()



