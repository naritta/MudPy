
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mfcc import Mfcc
from autoencoder import denosingAutoencoder

def denoise(learning_time=10,learning_rate=0.1):

    mfcc_dir1 = './mfcc1/'
    mfcc_dir2 = './mfcc2/'
    mfcc = Mfcc(mfcc_dir1, mfcc_dir2)
    data_x, data_y = mfcc.get_data()
    mfcc_dim = len(data_x[0])

    input_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    print '... building the model'

    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = denosingAutoencoder(
        np_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=mfcc_dim,
        n_hidden=5000
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
                    inputs=[],
                    outputs=cost,
                    updates=updates,
                    givens={
                        x: input_x,
                           }
                )

    ##########
    # TRAIN  #
    ##########

    print da.W.get_value()

    for i in range(learning_time):
        pred = train_da()

    # get get_reconstructed_input MFCC
    print len(da.get_hidden_values(input_x).eval()[0])
    print da.get_reconstructed_input(da.get_hidden_values(input_x)).eval()

if __name__ == '__main__':
    denoise()