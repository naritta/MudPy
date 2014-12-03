
import numpy as np

import theano
import theano.tensor as T

from mfcc import Mfcc

from multiplelayer import HiddenLayer
from logisticregression import LogisticRegression
from convolutional import ConvolutionalLayer

def evaluate_lenet5(learning_time=2, learning_rate=0.1,cnn_layer_units=[10, 20], batch_size=4,class_num=2):

    rng = np.random.RandomState(23455)

    mfcc_dir1 = './mfcc1/'
    mfcc_dir2 = './mfcc2/'
    mfcc = Mfcc(mfcc_dir1, mfcc_dir2)
    data_x, data_y = mfcc.get_data()
    mfcc_dim = mfcc.get_dim()
    mfcc_raw = mfcc.get_raw()

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

    layer0_input = x.reshape((batch_size, 1, mfcc_raw, mfcc_dim))

    filter_raw0 = 1002
    filter_dim0 = 11

    layer0 = ConvolutionalLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, mfcc_raw, mfcc_dim),
        filter_shape=(cnn_layer_units[0], 1, filter_raw0, filter_dim0),
        poolsize=(2, 2)
    )

    

    layer1 = ConvolutionalLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, cnn_layer_units[0], 1000, 5),
        filter_shape=(cnn_layer_units[1], cnn_layer_units[0], 201, 2),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(class_num)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=cnn_layer_units[1] * 400 * 2,
        n_out=10,
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=10, n_out=class_num)

    cost = layer3.negative_log_likelihood(y)

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: input_x,
                y: input_y}
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    for i in range(learning_time):
        pred = train_model()

    print layer0.W.eval()

if __name__ == '__main__':
    evaluate_lenet5()
