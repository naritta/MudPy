
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mfcc import Mfcc

class dA(object):

    def __init__(
        self,
        np_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


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

    da = dA(
        np_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=mfcc_dim,
        n_hidden=500
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
    print da.get_reconstructed_input(da.get_hidden_values(input_x)).eval()

if __name__ == '__main__':
    denoise()