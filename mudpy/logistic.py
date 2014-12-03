#coding:utf-8

import os
import struct

import numpy as np
import theano
import theano.tensor as T

from logisticregression import LogisticRegression
from mfcc import Mfcc

def calc_prediction(learning_rate=0.13, class_num=2):

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
    x = T.matrix("x")
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=mfcc_dim, n_out=class_num)

    cost = classifier.negative_log_likelihood(y)

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

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: input_x,
                y: input_y})

    ##########
    # TRAIN  #
    ##########

    for i in range(100):
        pred = train_model()

    ###############
    # prediction  #
    ###############

    # when predict all probability
    probability_model = theano.function(inputs=[index],
            outputs=classifier.p_y_given_x,
            givens={
                x: input_x[index:index+1]})

    # when predict argmax
    predict_model = theano.function(inputs=[index],
            outputs=classifier.y_pred,
            givens={
                x: input_x[index:index+1]})

    print "all probability is: "
    print probability_model(1)
    print "predicted digit is: "
    print predict_model(1)

    print classifier.W.get_value()

if __name__ == '__main__':
    calc_prediction()
