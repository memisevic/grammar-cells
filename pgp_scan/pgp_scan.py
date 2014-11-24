#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import shutil
import warnings

import numpy as np
import tables
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# debugging
DEBUG_MODE = False
#DEBUG_MODE = True
if DEBUG_MODE:
    theano.config.compute_test_value = 'raise'
    theano.config.optimizer = 'fast_compile'
    theano.config.exception_verbosity = 'high'

class LayerBase(object):
    def __init__(self, name):
        if type(self) == LayerBase:
            raise NotImplementedError('This base class should not be used directly')
        self.name = name

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def updateparams_fromdict(self, newparams):
        for p in self.params:
            try:
                p.set_value(newparams[p.name])
            except KeyError:
                print 'param {0} not in dict (will not be overwritten)'.format(
                    p.name)

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return np.concatenate(
            [p.get_value().flatten() for p in self.params])

    def save(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            print 'saving h5 file'
            self.save_h5(filename)
        elif ext == '.npy':
            print 'saving npy file'
            self.save_npy(filename)
        elif ext == '.npz':
            print 'saving npz file'
            self.save_npz(filename)
        else:
            print 'unknown file extension: {}'.format(ext)

    def save_h5(self, filename):
        try:
            shutil.copyfile(filename, '{}_bak'.format(filename))
        except IOError:
            print 'could not make backup of model param file (which is normal if we haven\'t saved one until now)'

        with tables.openFile(filename, 'w') as h5file:
            for p in self.params:
                h5file.createArray(h5file.root, p.name, p.get_value())
                h5file.flush()

    def save_npy(self, filename):
        np.save(filename, self.get_params())

    def save_npz(self, filename):
        np.savez(filename, **(self.get_params_dict()))

    def load_h5(self, filename):
        h5file = tables.openFile(filename, 'r')
        new_params = {}
        for p in h5file.listNodes(h5file.root):
            new_params[p.name] = p.read()
        self.updateparams_fromdict(new_params)
        h5file.close()


    def load(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            self.load_h5(filename)
        else:
            try:
                new_params = np.load(filename)
            except IOError, e:
                warnings.warn('''Parameter file could not be loaded with numpy.load()!
                            Is the filename correct?\n %s''' % (e, ))
            if type(new_params) == np.ndarray:
                print "loading npy file"
                self.updateparams(new_params)
            elif type(new_params) == np.lib.npyio.NpzFile:
                print "loading npz file"
                self.updateparams_fromdict(new_params)
            else:
                warnings.warn('''Parameter file loaded, but variable type not
                            recognized. Need npz or ndarray.''', Warning)


class GAE_Layer(LayerBase):

    @property
    def corruption_level(self):
        return self._corruption_level.get_value()

    @corruption_level.setter
    def corruption_level(self, val):
        self._corruption_level.set_value(np.float32(val))

    def __init__(self, nvis, nfac, nmap, name, input_type='real',
                 corruption_type='zeromask', corruption_level=.0,
                 numpy_rng=None, theano_rng=None, act_fn=T.nnet.sigmoid,
                 wxf_init=None, wyf_init=None):
        # call base class' constructor
        super(GAE_Layer, self).__init__(name=name)
        self.nvis = nvis
        self.nfac = nfac
        self.nmap = nmap
        self.input_type = input_type
        self.corruption_type = corruption_type
        self._corruption_level = corruption_level
        self.act_fn = act_fn

        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:
            self.theano_rng = RandomStreams(1)
        else:
            self.theano_rng = theano_rng

        if wxf_init is None:
            wxf_init = numpy_rng.uniform(low=-.005, high=.005,
                                        size=(nvis, nfac)).astype(
                                            theano.config.floatX)
        if wyf_init is None:
            wyf_init = numpy_rng.uniform(low=-.005, high=.005,
                                         size=(nvis, nfac)).astype(
                                            theano.config.floatX)

        self.wxf = theano.shared(wxf_init, name='wxf_{0}'.format(self.name))
        self.wyf = theano.shared(wyf_init, name='wyf_{0}'.format(self.name))

        self.wmf_init = numpy_rng.uniform(
            low=-.01, high=.01, size=(nmap, nfac)).astype(
                theano.config.floatX)
        self.wfm_init = numpy_rng.uniform(
            low=-.01, high=.01, size=(nfac, nmap)).astype(
                theano.config.floatX)
        self.wmf = theano.shared(value = self.wmf_init, name='wmf_{0}'.format(
            self.name))
        self.wfm = theano.shared(value = self.wfm_init, name='wfm_{0}'.format(
            self.name))

        self.bx = theano.shared(np.zeros(nvis, dtype=theano.config.floatX),
                                name='bx_{0}'.format(self.name))
        self.by = theano.shared(np.zeros(nvis, dtype=theano.config.floatX),
                                name='by_{0}'.format(self.name))
        self.bm = theano.shared(
            np.zeros(nmap, dtype=theano.config.floatX) - 2.,
            name='bm_{0}'.format(self.name))

        self.params = [self.wxf, self.wyf, self.wmf, self.wfm, self.bx, self.by,
                       self.bm]

        self._corruption_level = theano.shared(corruption_level)

        self.inputs = T.matrix(name='inputs_{0}'.format(self.name))
        self.inputs.tag.test_value = np.random.randn(
            100, nvis+nvis).astype(theano.config.floatX)

        self._inputsX = self.inputs[:, :nvis]
        self._inputsY = self.inputs[:, nvis:]

        if self.corruption_type=='zeromask':
            self._corruptedX = theano_rng.binomial(
                size=self._inputsX.shape, n=1, p=1.0-self._corruption_level,
                dtype=theano.config.floatX) * self._inputsX
            self._corruptedY = theano_rng.binomial(
                size=self._inputsY.shape, n=1, p=1.0-self._corruption_level,
                dtype=theano.config.floatX) * self._inputsY
        elif self.corruption_type is None:
            self._corruptedX = self._inputsX
            self._corruptedY = self._inputsY
        else:
            raise ValueError('unsupported noise type')

        self._factorsX = T.dot(self._corruptedX, self.wxf)
        self._factorsY = T.dot(self._corruptedY, self.wyf)
        self._preactMappings = T.dot(
            self._factorsX * self._factorsY, self.wfm) + self.bm
        self._mappings = self.act_fn(self._preactMappings)

        self._factorsM = T.dot(self._mappings, self.wmf)

        self._preactReconsX = T.dot(
            self._factorsY * self._factorsM, self.wxf.T) + self.bx
        self._preactReconsY = T.dot(
            self._factorsX * self._factorsM, self.wyf.T) + self.by

        if self.input_type == 'real':
            self._reconsX = self._preactReconsX
            self._reconsY= self._preactReconsY
            self._costpercase = T.sum(
                0.5*((self._inputsX-self._reconsX)**2) +
                0.5*((self._inputsY-self._reconsY)**2), axis=1)
        elif self.input_type == 'binary':
            self._reconsX = T.nnet.sigmoid(self._preactReconsX)
            self._reconsY = T.nnet.sigmoid(self._preactReconsY)
            self._costpercase = - T.sum(
                0.5* (self._inputsY*T.log(self._reconsY) +
                      (1.0-self._inputsY)*T.log(1.0-self._reconsY)) +
                0.5* (self._inputsX*T.log(self._reconsX) +
                      (1.0-self._inputsX)*T.log(1.0-self._reconsX)),
                axis=1)
        else:
            raise Value('unsupported output type')

        self._cost = T.mean(self._costpercase)
        self._grads = T.grad(self._cost, self.params)

        self.pMGivenXY = theano.OpFromGraph(
            [self._inputsX, self._inputsY], [self._preactMappings])

        self.pXGivenMY = theano.OpFromGraph(
            [self._mappings, self._inputsY], [self._preactReconsX])

        self.pYGivenMX = theano.OpFromGraph(
            [self._mappings, self._inputsX], [self._preactReconsY])

        self.mappings = theano.function(
            [self.inputs], self.act_fn(
                self.pMGivenXY(self._inputsX, self._inputsY)))



class PGP(object):
    """Predictive Gating Pyramid"""

    def __init__(self, nvis, nnote, nfacs, nmaps, input_type='real',
                 corruption_types=None, corruption_levels=None,
                 numpy_rng=None, theano_rng=None, autonomy_init=.5,
                 act_fns=None):

        assert len(nfacs) == len(nmaps), (
            'nfacs has to be of same length as nmaps')
        self.nlayers = len(nfacs)
        assert self.nlayers > 0, (
            'specify number of factors and mappings for at least one layer!')
        if corruption_types is None:
            self.corruption_types = [None] * self.nlayers
        else:
            self.corruption_types = corruption_types
        if corruption_levels is None:
            corruption_levels = [.0] * self.nlayers
        else:
            corruption_levels = corruption_levels

        if act_fns is None:
            self.act_fns = [T.nnet.sigmoid] * self.nlayers
        else:
            self.act_fns = act_fns


        self.nvis = nvis
        self.nnote = nnote
        self.nfacs = nfacs
        self.nmaps = nmaps
        self.input_type = input_type

        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:
            self.theano_rng = RandomStreams(1)
        else:
            self.theano_rng = theano_rng


        self.autonomy = theano.shared(
            value=np.float32(autonomy_init), name='autonomy')
        self.inputs = T.matrix(name='inputs')
        self.inputs.tag.test_value = np.random.randn(
            100, 20*self.nvis).astype(np.float32)

        # reshape inputs to 3d tensor with axis order time, batch, framesize
        # (scan loops over first axis)
        self._inputs = self.inputs.reshape(
            (self.inputs.shape[0], self.inputs.shape[1] / nvis,
             nvis)).dimshuffle(1,0,2)

        # if memory units should be used, augment initial input vectors with
        # small random numbers
        if nnote > 0:
            self._inputs_w_note = T.concatenate((
                self._inputs, theano_rng.uniform(
                    low=-.00001, high=.00001,
                    size=(self._inputs.shape[0], self._inputs.shape[1], nnote))),
                axis=2)
        else:
            self._inputs_w_note = self._inputs

        self.layers = [GAE_Layer(
            nvis=nvis+nnote, nfac=nfacs[0], nmap=nmaps[0], name='Layer0_GAE',
            input_type=input_type, corruption_type=self.corruption_types[0],
            corruption_level=corruption_levels[0], numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng, act_fn=self.act_fns[0])]
        self.params = filter(lambda x: x.name != 'bx_{0}'.format(
            self.layers[0].name), self.layers[0].params)

        self.corruption_levels = [self.layers[0].corruption_level]

        for l in xrange(1, self.nlayers):
            self.layers.append(GAE_Layer(
                nvis=nmaps[l-1], nfac=nfacs[l], nmap=nmaps[l],
                name='Layer{0}_GAE'.format(l), input_type='binary',
                corruption_type=self.corruption_types[l],
                corruption_level=corruption_levels[l],
                numpy_rng=self.numpy_rng, theano_rng=self.theano_rng,
                act_fn=self.act_fns[l]))

            self.params.extend(filter(lambda x: x.name != 'bx_{0}'.format(
                self.layers[-1].name), self.layers[-1].params))

            self.corruption_levels.append(self.layers[l].corruption_level)

        def step(x_tp1, inputs, pmTop_t):
            # FIXME: pre-activation horizontally
            inps = T.concatenate((inputs, x_tp1.dimshuffle('x',0,1)), axis=0)

            mappings = []
            # infer first layer mappings
            mappings.append([])
            for i in xrange(0, self.nlayers):
                mappings[0].append(self.layers[0].pMGivenXY(inps[i], inps[i+1]))

            # infer intermediate layer mappings
            for l in xrange(1, self.nlayers):
                mappings.append([])
                for i in xrange(self.nlayers-l):
                    mappings[l].append(self.layers[l].pMGivenXY(
                        self.layers[l-1].act_fn(mappings[l-1][i]),
                        self.layers[l-1].act_fn(mappings[l-1][i+1])))

            # average top layer mappings
            mappings[-1][0] = self.autonomy * pmTop_t + (
                1 - self.autonomy) * mappings[-1][0]
            mappings[-1][0].name = 'top-layer-maps'

            # treat top-layer mappings as "prediction from top+1-th layer"
            preds = [mappings[-1][0]]

            for l in xrange(self.nlayers-1, 0, -1):
                preds.append(self.layers[l].pYGivenMX(
                    self.layers[l].act_fn(preds[-1]), mappings[l-1][-2]))

                # average top-down prediction with bottom-up inference based on
                # autonomy weight (autonomy=1. means only prediction
                preds[-1] = self.autonomy * preds[-1] + (
                    1 - self.autonomy) * mappings[l-1][-1]

            # compute input prediction
            preds.append(self.layers[0].pYGivenMX(
                preds[-1], inps[-2]))

            if self.input_type == 'real':
                stepCost = T.mean((preds[-1] - x_tp1)**2)
            else:
                raise ValueError('unsupported input type')
            new_inputs = T.concatenate((inputs[1:], preds[-1].dimshuffle('x',0,1)), axis=0)

            return new_inputs, mappings[-1][0], stepCost

        inps = self._inputs_w_note[:self.nlayers + 1]

        mappings = []
        # infer first layer mappings
        mappings.append([])
        for i in xrange(0, self.nlayers):
            mappings[0].append(self.layers[0].pMGivenXY(inps[i], inps[i+1]))
            # FIXME: need to apply sigmoid before passing up

        # infer intermediate layer mappings
        for l in xrange(1, self.nlayers):
            mappings.append([])
            for i in xrange(self.nlayers-l):
                mappings[l].append(self.layers[l].pMGivenXY(
                    mappings[l-1][i], mappings[l-1][i+1]))

        pmTop_init = mappings[-1][0]

        (_seqPlusPreds, _topMappings, _stepCosts), _ = theano.scan(
            fn=step,
            sequences=self._inputs_w_note[self.nlayers + 1:],
            outputs_info=[T.concatenate([
                self._inputs_w_note[i:i+1] for i in range(1, self.nlayers + 1)],
                axis=0),
                pmTop_init, None])

        self._cost = T.mean(_stepCosts)
        self._grads = T.grad(self._cost, self.params)
        self._predictions = _seqPlusPreds[:, -1]

        self.nsteps = T.lscalar('nsteps')
        self.nsteps.tag.test_value = 10
        givens = {}
        for l in self.layers:
            givens[l._corruption_level] = .0
        givens[self._inputs_w_note] = T.concatenate((
            self._inputs_w_note[:self.nlayers + 1],
            T.zeros((self.nsteps, self._inputs_w_note.shape[1],
                     self.nvis + self.nnote), dtype=theano.config.floatX)),
            axis=0)

        self.predict = theano.function(
            [self.inputs, self.nsteps],
            self._predictions.dimshuffle(1,0,2).flatten(2),
            givens=givens)

    def save(self, fname_pattern):
        """Saves all layers' parameters

        Args
        ----
            fname_pattern: a string with a place holder specifying the filename
                pattern, e.g. "path/to/bak/layer_{0}.h5", {0} is then replaced
                by the layers name.
        """
        for layer in self.layers:
            dirname, filename = os.path.split(fname_pattern)
            if dirname != '':
                os.system('mkdir -p {0}'.format(dirname))
            layer.save(fname_pattern.format(layer.name))

    def load(self, fname_pattern):
        """Loads all layers' parameters

        Args
        ----
            fname_pattern: a string with a place holder specifying the filename
                pattern, e.g. "path/to/bak/layer_{0}.h5", {0} is then replaced
                by the layers name.
        """
        for layer in self.layers:
            dirname, filename = os.path.split(fname_pattern)
            layer.load(fname_pattern.format(layer.name))


if __name__ == '__main__':
    # there seems to be a bug in this code or in theano (?), the OpFromGraph
    # node inside scan seems to see some unused inputs (which were not
    # added manually as far as i can tell). As a workaround the following line
    # can be uncommented:
    theano.config.on_unused_input = 'warn'

    from graddescent_rewrite import SGD_Trainer

    TRAIN = True
    PRETRAIN = False
    numpy_rng = np.random.RandomState(1)

    print 'making data...'
    seq_len = 160
    frame_len = 10
    nframes = seq_len / frame_len
    ntrain = 10000

    train_features_numpy = 1000*np.array(
        [np.sin(np.linspace(o, o+2*np.pi, seq_len)*m) for m, o in zip(
            numpy_rng.rand(ntrain)*29+1, numpy_rng.rand(
                ntrain)*2*np.pi)]).astype("float32")

    data_mean = train_features_numpy.mean()
    train_features_numpy -= data_mean
    data_std = train_features_numpy.std()
    train_features_numpy /= data_std
    train_features_numpy = train_features_numpy[np.random.permutation(ntrain)]
    train_features_theano = theano.shared(train_features_numpy)

    print 'instantiating model...'
    testmodel = PGP(nvis=frame_len, nnote=0, nfacs=[200,100],#,10],
                    nmaps=[100,50],#,10],
                    input_type='real',
                    corruption_types=None,
                    #corruption_types=['zeromask', 'zeromask', 'zeromask'],
                    corruption_levels=[.3, .3],#, .3],
                    numpy_rng=numpy_rng, theano_rng=None,
                    act_fns=None)

    if TRAIN:
        print 'pretraining first layer model...'

        if PRETRAIN:
            pretrain_features_velocity = np.concatenate(
                [train_features_numpy[i, 2*j*frame_len:2*(j+1)*frame_len][None,:]
                for j in range(seq_len/(frame_len*2)) for i in range(ntrain)],0)

            pretrainer1 = SGD_Trainer(
                model=testmodel.layers[0],
                inputs=pretrain_features_velocity,
                batchsize=100,
                learningrate=0.001,
                loadsize=10000)

            for epoch in xrange(50):
                pretrainer1.step()

            print 'pretraining second layer model...'
            pretrain_features_acceleration_numpy = np.concatenate((
                testmodel.layers[0].mappings(
                    train_features_numpy[:, :2*frame_len]),
                testmodel.layers[0].mappings(
                    train_features_numpy[:, 1*frame_len:3*frame_len])),1)

            pretrainer2 = SGD_Trainer(
                model=testmodel.layers[1],
                inputs=pretrain_features_acceleration_numpy,
                batchsize=100,
                learningrate=0.001,
                loadsize=10000)

            for epoch in xrange(50):
                pretrainer2.step()

        else:
            testmodel.load('test/sine_model_{0}.h5')

        print 'training PGP...'
        trainer = SGD_Trainer(
            model=testmodel,
            inputs=train_features_numpy[:, :frame_len*4],
            batchsize=100,
            learningrate=.01,
            loadsize=10000)

        # train with increasing number of prediction step, each time for 100
        # epochs
        for nsteps in range(1, 10):
            print 'training PGP with {0}-step training'.format(nsteps)
            trainer.inputs = train_features_numpy[:, :frame_len*(testmodel.nlayers + nsteps + 1)]
            print trainer._inputs_theano.get_value().shape
            for epoch in xrange(100):
                trainer.step()

            testmodel.save('test_tmp/sine_model_{0}.h5')

        testmodel.save('test/sine_model_{0}.h5')
    else:
        testmodel.load('test/sine_model_{0}.h5')


    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(train_features_numpy[0])
    plt.plot(testmodel.predict(
        train_features_numpy[[0]], nframes).flatten())
    plt.show(); plt.draw()
    plt.savefig('testplot.png')

# vim: set ts=4 sw=4 sts=4 expandtab:
