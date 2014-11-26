
import numpy
import theano
import theano.tensor as T

rng = numpy.random

N = 400
feats = 784

dim = 3

x_input = numpy.array([[[1,2],[2,1]],[[3,3],[1,1]],[[1,-2],[2,-3]],[[2,-1],[3,-3]]])
teach = numpy.array([1,1,0,0])
training_steps = 10000

x = T.btensor3("x")
y = T.vector("y")
w = theano.shared(rng.randn(dim), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)
test = theano.function(inputs=[x,y], outputs=cost)


for i in range(training_steps):
    pred, err = train([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],[2,1])

print predict([[[1,2,3]],[[1,2,3]]])