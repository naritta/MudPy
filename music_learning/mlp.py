
import os
import struct

import numpy as np

import theano
import theano.tensor as T

from multiplelayer import MLP

def load_mfcc(mfcc_file, m):

    mfcc = []
    fp = open(mfcc_file, "rb")
    while True:
        b = fp.read(4)
        if b == "": break
        val = struct.unpack("f", b)[0]
        mfcc.append(val)
    fp.close()

    mfcc = np.array(mfcc)
    num_frame = len(mfcc) / m
    mfcc = mfcc.reshape(num_frame, m)

    return mfcc

def create_mfcc(mfccDir):

    all_mfcc_1d = []

    for i,file in enumerate(os.listdir(mfccDir)):
        if not file.endswith(".mfc"): continue
        mfccFile = os.path.join(mfccDir, file)

        mfcc = load_mfcc(mfccFile, 20)
        mfcc_1d = []
        for i in xrange(len(mfcc)):
            mfcc_1d.extend(mfcc[i])

        all_mfcc_1d.append(mfcc_1d)

    return all_mfcc_1d

def make_data(mfcc_dir1, mfcc_dir2):

    mfccs1 = create_mfcc(mfcc_dir1)
    num_mfccs1 = len(mfccs1)
    classes1 = [ 0 for _ in xrange(num_mfccs1)]

    mfccs2 = create_mfcc(mfcc_dir2)
    num_mfccs2 = len(mfccs2)
    classes2 = [ 1 for _ in xrange(num_mfccs2)]

    data_x=[]
    data_x.extend(mfccs1)
    data_x.extend(mfccs2)
    data_y=[]
    data_y.extend(classes1)
    data_y.extend(classes2)

    return data_x, data_y

def calc_prediction(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_hidden=500):

    mfcc_dir1 = './mfcc1/'
    mfcc_dir2 = './mfcc2/'
    data_x, data_y = make_data(mfcc_dir1, mfcc_dir2)
    mfcc_dim = len(data_x[0])

    input_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
    input_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=True)
    input_y = T.cast(input_y, 'int32')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=mfcc_dim,
        n_hidden=n_hidden,
        n_out=2
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: input_x,
                y: input_y})

    validate_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: input_x,
                y: input_y})

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: input_x,
                y: input_y})

    ##########
    # TRAIN  #
    ##########

    for i in range(10):
        pred = train_model()


    ###############
    # prediction  #
    ###############

    # when predict all probability
    probability_model = theano.function(inputs=[index],
            outputs=classifier.logRegressionLayer.p_y_given_x,
            givens={
                x: input_x[index:index+1]})

    # when predict argmax
    predict_model = theano.function(inputs=[index],
            outputs=classifier.logRegressionLayer.y_pred,
            givens={
                x: input_x[index:index+1]})

    print "all probability is: "
    print probability_model(1)
    print "predicted digit is: "
    print predict_model(1)

if __name__ == '__main__':
    calc_prediction()