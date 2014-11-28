
import numpy as np

import theano
import theano.tensor as T

from multiplelayer import MLP
from mfcc import Mfcc

def calc_prediction(learning_time=10, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_hidden=500):

    mfcc_dir1 = './mfcc1/'
    mfcc_dir2 = './mfcc2/'
    mfcc = Mfcc(mfcc_dir1, mfcc_dir2)
    data_x, data_y = mfcc.get_data()
    mfcc_dim = len(data_x[0])

    input_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
    input_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=True)
    input_y = T.cast(input_y, 'int32')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

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

    for i in range(learning_time):
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