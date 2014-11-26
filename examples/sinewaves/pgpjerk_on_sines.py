### Train grammar-cells on sine waves 

import pylab
import numpy
import numpy.random
import gatedAutoencoder
import pgp3layer 
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    from pylab import cm, ceil
    numimages = M.shape[1]
    if layout is None:
        n0 = int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=cm.gray, interpolation='nearest', **kwargs)
    pylab.draw(); pylab.show()


class GraddescentMinibatch(object):
    """ Gradient descent trainer class. """

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, normalizefilters=False, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = theano.shared(numpy.array(learningrate).astype("float32"))
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        self.normalizefilters = normalizefilters 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            if self.normalizefilters:
                self.model.normalizefilters()

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

    def reset_incs(self):
        for p in self.model.params:
            self.incs[p].set_value(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))




print 'making data...'
seq_len = 160 
frame_len = 10
numframes = seq_len / frame_len
numtrain = 50000
numtest = 10000

#all_features_numpy = 1000*numpy.array([numpy.sin(numpy.linspace(o, o+2*numpy.pi, seq_len)*m) for m, o in zip(numpy_rng.rand(numtrain+numtest)*580+220, numpy_rng.rand(numtrain+numtest)*2*numpy.pi)]).astype("float32")
all_features_numpy = 1000*numpy.array([numpy.sin(numpy.linspace(o, o+2*numpy.pi, seq_len)*m) for m, o in zip(numpy_rng.rand(numtrain+numtest)*29+1, numpy_rng.rand(numtrain+numtest)*2*numpy.pi)]).astype("float32")
train_features_numpy = all_features_numpy[:numtrain]
test_features = all_features_numpy[numtrain:]
print all_features_numpy.shape
del all_features_numpy

data_mean = train_features_numpy.mean()
train_features_numpy -= data_mean
data_std = train_features_numpy.std()
train_features_numpy /= data_std 
train_features_numpy = train_features_numpy[numpy.random.permutation(numtrain)]
train_features_theano = theano.shared(train_features_numpy)
test_features -= data_mean
test_features /= data_std
test_feature_beginnings = test_features[:,:frame_len*3]
print '... done'




print 'pretraining velocity model ...'
pretrainmodel_velocity = gatedAutoencoder.FactoredGatedAutoencoder(
                                                          numvisX=frame_len,
                                                          numvisY=frame_len,
                                                          numfac=200, 
                                                          nummap=100, 
                                                          output_type='real', 
                                                          corruption_type='zeromask', 
                                                          corruption_level=0.3, 
                                                          numpy_rng=numpy_rng, 
                                                          theano_rng=theano_rng)

pretrain_features_velocity_numpy = numpy.concatenate([train_features_numpy[i, 2*j*frame_len:2*(j+1)*frame_len][None,:] for j in range(seq_len/(frame_len*2)) for i in range(numtrain)],0)
pretrain_features_velocity_numpy = pretrain_features_velocity_numpy[numpy.random.permutation(pretrain_features_velocity_numpy.shape[0])]
pretrain_features_velocity_theano = theano.shared(pretrain_features_velocity_numpy)
pretrainer_velocity = GraddescentMinibatch(pretrainmodel_velocity, pretrain_features_velocity_theano, batchsize=100, learningrate=0.01)
for epoch in xrange(1):
    pretrainer_velocity.step()

print '... done'


print 'pretraining acceleration model ...'
pretrainmodel_acceleration = gatedAutoencoder.FactoredGatedAutoencoder(
                                                          numvisX=pretrainmodel_velocity.nummap,
                                                          numvisY=pretrainmodel_velocity.nummap,
                                                          numfac=100, 
                                                          nummap=50, 
                                                          output_type='real', 
                                                          corruption_type='zeromask', 
                                                          corruption_level=0.3, 
                                                          numpy_rng=numpy_rng, 
                                                          theano_rng=theano_rng)

pretrain_features_acceleration_numpy = numpy.concatenate((pretrainmodel_velocity.mappings(train_features_numpy[:, :2*frame_len]), 
                                                          pretrainmodel_velocity.mappings(train_features_numpy[:, 1*frame_len:3*frame_len])),1)
pretrain_features_acceleration_theano = theano.shared(pretrain_features_acceleration_numpy)
pretrainer_acceleration = GraddescentMinibatch(pretrainmodel_acceleration, pretrain_features_acceleration_theano, batchsize=100, learningrate=0.01)
for epoch in xrange(1):
    pretrainer_acceleration.step()
    #pylab.imshow(pretrainmodel_acceleration.mappings(pretrain_features_acceleration_numpy[:200]))
    #pylab.show(); pylab.draw()

print '... done'


print 'training sequence model ...'
model = pgp3layer.Pgp3layer(numvis=frame_len,
                            numnote=0,
                            numfac=200,
                            numvel=100,
                            numvelfac=100,
                            numacc=50,
                            numaccfac=10,
                            numjerk=10,
                            output_type='vocoder',
                            vis_corruption_type='zeromask',
                            vis_corruption_level=0.0,
                            numframes=numframes,
                            numpy_rng=numpy_rng,
                            theano_rng=theano_rng)


model.wxf_left.set_value(numpy.concatenate((pretrainmodel_velocity.wxf.get_value()*0.5, numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))
model.wxf_right.set_value(numpy.concatenate((pretrainmodel_velocity.wyf.get_value()*0.5, numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))
model.wv.set_value(pretrainmodel_velocity.whf.get_value().T)
model.wvf_left.set_value(pretrainmodel_acceleration.wxf.get_value())
model.wvf_right.set_value(pretrainmodel_acceleration.wyf.get_value())
model.wa.set_value(pretrainmodel_acceleration.whf.get_value().T)
model.ba.set_value(pretrainmodel_acceleration.bmap.get_value())
model.bv.set_value(pretrainmodel_velocity.bmap.get_value())
model.bx.set_value(numpy.concatenate((pretrainmodel_velocity.bvisX.get_value(),numpy.zeros((model.numnote),dtype="float32"))))
model.autonomy.set_value(numpy.array([0.5], dtype="float32"))


# TRAIN MODEL
trainer = GraddescentMinibatch(model, train_features_theano, batchsize=400, learningrate=0.001)
for epoch in xrange(500):
    trainer.step()
    #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
    #plot(train_features_numpy[0].flatten())
    #plot(model.predict(train_features_numpy[[0]], 100).flatten())

model.vis_corruption_level.set_value(numpy.array([0.5]).astype("float32"))
for epoch in xrange(500):
    trainer.step()
    #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

model.vis_corruption_level.set_value(numpy.array([0.99]).astype("float32"))
for epoch in xrange(1000):
    trainer.step()
    #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

print '... done'

#plot(train_features_numpy[0].flatten())
#plot(model.predict(train_features_numpy[[0]], 100).flatten())


