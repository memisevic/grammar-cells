import collections
import cPickle as pickle
import os
import shutil
import warnings

import numpy as np
import theano
import theano.tensor as T
import tables
#theano.config.compute_test_value = 'warn'


class SGD_Trainer(object):
    """Implementation of a stochastic gradient descent trainer
    """

#{{{ Properties

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, val):
        #FIXME: make this work for other input types
        if not isinstance(val, np.ndarray):
            raise TypeError('Resetting trainer inputs currently only works for '
                        'ndarray inputs!')
        self._inputs = val
        self._inputs_theano = theano.shared(
            self._inputs[:self._loadsize],
            name='inputs')
        self._numcases = self._inputs.shape[0]
        self._numloads = self._numcases // self._loadsize

    @property
    def gradient_clip_threshold(self):
        return self._gradient_clip_threshold.get_value()

    @property
    def learningrate_decay_factor(self):
        return self._learningrate_decay_factor.get_value()

    @learningrate_decay_factor.setter
    def learningrate_decay_factor(self, val):
        self._learningrate_decay_factor.set_value(np.float32(val))

    @property
    def learningrate_decay_interval(self):
        return self._learningrate_decay_interval.get_value()

    @learningrate_decay_interval.setter
    def learningrate_decay_interval(self, val):
        self._learningrate_decay_interval.set_value(np.int64(val))

    @gradient_clip_threshold.setter
    def gradient_clip_threshold(self, val):
        self._gradient_clip_threshold.set_value(np.float32(val))

    @property
    def learningrate(self):
        return self._learningrate.get_value()

    @learningrate.setter
    def learningrate(self, value):
        self._learningrate.set_value(np.float32(value))

    @property
    def momentum(self):
        return self._momentum.get_value()

    @momentum.setter
    def momentum(self, val):
        self._momentum.set_value(np.float32(val))

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def loadsize(self):
        return self._loadsize

    @property
    def numcases(self):
        return self._numcases

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = bool(val)

    @property
    def epochcount(self):
        return self._epochcount

    @epochcount.setter
    def epochcount(self, val):
        self._epochcount = int(val)

    @property
    def momentum_batchcounter(self):
        return self._momentum_batchcounter
#}}}

    def __init__(self, model, inputs, batchsize, learningrate,
                 momentum=0.9, loadsize=None,
                 rng=None, verbose=True,
                 numcases=None, gradient_clip_threshold=1000,
                 valid_inputs=None,
                 patience=None, patience_increase=None,
                 improvement_threshold=None,
                 validation_frequency=None, numepochs_per_load=1):

#{{{ Initialization of Properties
        self._learningrate = theano.shared(np.float32(learningrate),
                                           name='learningrate')
        self.numepochs_per_load = numepochs_per_load

        self._momentum = theano.shared(np.float32(momentum),
                                       name='momentum')
        self._total_stepcount = 0

        self._gradient_clip_threshold = theano.shared(
                np.float32(gradient_clip_threshold),
                name='gradient_clip_threshold')
        self._avg_gradnorm = theano.shared(np.float32(0.), name='avg_gradnorm')

        self._learningrate_decay_factor = theano.shared(
            np.float32,
            name='learningrate_decay_factor')

        self._learningrate_decay_interval = theano.shared(
            np.int64,
            name='learningrate_decay_interval')

        if isinstance(inputs, str):
            self._inputs_type = 'h5'
            self._inputsfile = tables.openFile(inputs, 'r')
            self._inputs = self._inputsfile.root.inputs_white
        elif hasattr(inputs, '__call__'):
            self._inputs_type = 'function'
            self._inputs_fn = inputs
        else:
            self._inputs_type = 'numpy'
            self._inputs = inputs

        self._model = model

        self._numparams = reduce(lambda x,y: x+y,
            [p.get_value().size for p in self._model.params])

        if valid_inputs is None:
            self._valid_inputs = valid_inputs
        else:
            if isinstance(valid_inputs, str):
                self._valid_inputs_type = 'h5'
                self._valid_inputsfile = tables.openFile(valid_inputs, 'r')
                self._valid_inputs = self._valid_inputsfile.root.inputs_white
            else:
                self._valid_inputs_type = 'numpy'
                self._valid_inputs = valid_inputs
            self._valid_inputs_theano = theano.shared(
                self._valid_inputs)
            self._best_valid_params = \
            dict([(p.name, np.zeros(p.get_value().shape,
                                dtype=theano.config.floatX))
                                for p in self._model.params])
            self._best_valid_cost = np.infty
            self._patience = patience
            if patience_increase is None:
                print 'WARNING: using default value of 1.3 for patience increase'
                patience_increase = 1.3
            self._patience_increase = patience_increase
            if improvement_threshold is None:
                print 'WARNING: using default value of 0.98 for improvement threshold'
                improvement_threshold= 0.98
            self._improvement_threshold = improvement_threshold
            if validation_frequency is None:
                validation_frequency = min(self._inputs.shape[0]/batchsize, self._patience/2)
            self._validation_frequency = validation_frequency


        if self._inputs_type == 'function':
            numcases = loadsize
        else:
            if numcases is None or numcases > self._inputs.shape[0]:
                numcases = self._inputs.shape[0]
        self._numcases = numcases

        self._batchsize = batchsize
        self._loadsize = loadsize
        self._verbose       = verbose
        if self._batchsize > self._numcases:
            self._batchsize = self._numcases
        if self._loadsize == None:
            self._loadsize = self._batchsize * 100
        if self._loadsize > self._numcases:
            self._loadsize = self._numcases
        self._numloads      = self._numcases // self._loadsize
        self._numbatches    = self._loadsize // self._batchsize

        if self._inputs_type == 'h5':
            self._inputs_theano = theano.shared(
                self._inputs.read(stop=self._loadsize))
        elif self._inputs_type == 'function':
            # TODO: generate inputs for first load
            print "generating first load..."
            inp = np.empty((self._loadsize, ) + (self._inputs_fn().shape),
                           dtype=np.float32)
            for i in xrange(self._loadsize):
                inp[i] = self._inputs_fn()
                if (i + 1) % 100 == 0:
                    print '{0}/{1}'.format(i + 1, self.loadsize)

            self._inputs_theano = theano.shared(
                inp)
        else:
            self._inputs_theano = theano.shared(
                self._inputs[:self._loadsize],
                name='inputs')
        #self._inputs_theano.tag.test_value = np.random.randn(100, model.n_vis*4)

        self._momentum_batchcounter = 0

        if rng is None:
            self._rng = np.random.RandomState(1)
        else:
            self._rng = rng

        self._epochcount = 0
        self.epoch_of_last_valid_improvement = 0
        self._index = T.lscalar()
        self._incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self._model.params])
        self._inc_updates = collections.OrderedDict()
        self._updates_nomomentum = collections.OrderedDict()
        self._updates = collections.OrderedDict()
        self._n = T.lscalar('n')
        self._n.tag.test_value = 0.
        self._noop = 0.0 * self._n
        self._batch_idx = theano.shared(
            value=np.array(0, dtype=np.int64), name='batch_idx')
        self._compile_functions()

#}}}

    def __del__(self):
        if self._inputs_type == 'h5':
            self._inputsfile.close()
        if self._valid_inputs is not None:
            if self._valid_inputs_type == 'h5':
                self._valid_inputsfile.close()

    def save(self, filename):
        """Saves the trainers parameters to a file
        Params:
            filename: path to the file
        """
        ext = os.path.splitext(filename)[1]
        if ext == '.pkl':
            print 'saving trainer params to a pkl file'
            self.save_pkl(filename)
        else:
            print 'saving trainer params to a hdf5 file'
            self.save_h5(filename)

    def save_h5(self, filename):
        """Saves a HDF5 file containing the trainers parameters
        Params:
            filename: path to the file
        """
        try:
            shutil.copyfile(filename, '{0}_bak'.format(filename))
        except IOError:
            print 'could not make backup of trainer param file (which is \
                    normal if we haven\'t saved one until now)'
        paramfile = tables.openFile(filename, 'w')
        paramfile.createArray(paramfile.root, 'learningrate',
                              self.learningrate)
        paramfile.createArray(paramfile.root, 'verbose', self.verbose)
        paramfile.createArray(paramfile.root, 'loadsize', self.loadsize)
        paramfile.createArray(paramfile.root, 'batchsize', self.batchsize)
        paramfile.createArray(paramfile.root, 'momentum',
                              self.momentum)
        paramfile.createArray(paramfile.root, 'epochcount',
                              self.epochcount)
        paramfile.createArray(paramfile.root, 'momentum_batchcounter',
                              self.momentum_batchcounter)
        incsgrp = paramfile.createGroup(paramfile.root, 'incs', 'increments')
        for p in self._model.params:
            paramfile.createArray(incsgrp, p.name, self._incs[p].get_value())
        paramfile.close()

    def save_pkl(self, filename):
        """Saves a pickled dictionary containing the parameters to a file
        Params:
            filename: path to the file
        """
        param_dict = {}
        param_dict['learningrate'] = self.learningrate
        param_dict['verbose'] = self.verbose
        param_dict['loadsize'] = self.loadsize
        param_dict['batchsize'] = self.batchsize
        param_dict['momentum'] = self.momentum
        param_dict['epochcount'] = self.epochcount
        param_dict['momentum_batchcounter'] = self.momentum_batchcounter
        param_dict['incs'] = dict(
            [(p.name, self._incs[p].get_value()) for p in self._model.params])
        pickle.dump(param_dict, open(filename, 'wb'))

    def load(self, filename):
        """Loads pickled dictionary containing parameters from a file
        Params:
            filename: path to the file
        """
        param_dict = pickle.load(open('%s' % filename, 'rb'))
        self.learningrate = param_dict['learningrate']
        self.verbose = param_dict['verbose']
        self._loadsize = param_dict['loadsize']
        self._batchsize = param_dict['batchsize']
        self.momentum = param_dict['momentum']
        self.epochcount = param_dict['epochcount']
        self._momentum_batchcounter = param_dict['momentum_batchcounter']
        for param_name in param_dict['incs'].keys():
            for p in self._model.params:
                if p.name == param_name:
                    self._incs[p].set_value(param_dict['incs'][param_name])
        self._numbatches = self._loadsize // self._batchsize
        if self._inputs_type != 'function':
            self._numloads = self._inputs.shape[0] // self._loadsize
        if self._inputs_type == 'h5':
            self._inputs_theano.set_value(
                self._inputs.read(stop=self._loadsize))
        else:
            self._inputs_theano.set_value(self._inputs[:self._loadsize])

    def reset_incs(self):
        for p in self._model.params:
            self._incs[p].set_value(
                np.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def _compile_functions(self):
        _gradnorm = T.zeros([])
        for _grad in self._model._grads:
            _gradnorm += T.sum(_grad**2)
        self._gradnorm = T.sqrt(_gradnorm)
        self.gradnorm = theano.function(
            inputs=[],
            outputs=_gradnorm,
            givens={
                self._model.inputs:
                self._inputs_theano[
                    self._batch_idx*self.batchsize:
                    (self._batch_idx+1)*self.batchsize]})

        avg_gradnorm_update = {
            self._avg_gradnorm: self._avg_gradnorm * .8 + self._gradnorm * .2}

        for _param, _grad in zip(self._model.params, self._model._grads):
            if hasattr(self._model, 'skip_params'):
                if _param.name in self._model.skip_params:
                    continue
            _clip_grad = T.switch(
                T.gt(_gradnorm, self._gradient_clip_threshold),
                _grad * T.switch(
                    T.gt(self._avg_gradnorm, 1.),
                    self._avg_gradnorm, 1.) / _gradnorm, _grad)

            self._inc_updates[self._incs[_param]] = \
                    self._momentum * self._incs[_param] - \
                    self._learningrate * _grad
            self._updates[_param] = _param + self._incs[_param]
            self._updates_nomomentum[_param] = \
                    _param - self._learningrate * _clip_grad

            try:
                # Cliphid version:
                self._inc_updates[self._incs[_param]] = \
                        self._momentum * self._incs[_param] - \
                        self._learningrate * \
                        self._model.layer.learningrate_modifiers[
                            _param.name] * _clip_grad
                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                    self._learningrate * \
                    self._model.layer.learningrate_modifiers[_param.name] * \
                        _clip_grad
            except AttributeError:
                self._inc_updates[self._incs[_param]] = self._momentum * \
                        self._incs[_param] - self._learningrate * _clip_grad
                self._updates[_param] = _param + self._incs[_param]
                self._updates_nomomentum[_param] = _param - \
                        self._learningrate * _clip_grad

        # first update gradient norm running avg
        ordered_updates = collections.OrderedDict(avg_gradnorm_update)
        # so that it is considered in the parameter update computations
        ordered_updates.update(self._inc_updates)
        self._updateincs = theano.function(
            [], [self._model._cost, self._avg_gradnorm], updates = ordered_updates,
            givens = {self._model.inputs:self._inputs_theano[
                self._batch_idx*self._batchsize:(self._batch_idx+1)* \
                self._batchsize]})

        self._trainmodel = theano.function(
            [self._n], self._noop, updates = self._updates)

        self._trainmodel_nomomentum = theano.function(
            [self._n], self._noop, updates = self._updates_nomomentum,
            givens = {self._model.inputs:self._inputs_theano[
                self._batch_idx*self._batchsize:(self._batch_idx+1)* \
                self._batchsize]})
        if self._valid_inputs is not None:
            self._compute_validcost = theano.function(
                [self._n], self._model._validcost,
                givens = {self._model.inputs:
                        self._valid_inputs_theano[
                            self._n*self._batchsize:(self._n+1)*
                            self._batchsize]})

        self._momentum_batchcounter = 0

    def _trainsubstep(self, batchidx):
        self._batch_idx.set_value(batchidx)
        stepcost, avg_gradnorm = self._updateincs()


        if self._momentum_batchcounter < 10:
            self._momentum_batchcounter += 1
            self._trainmodel_nomomentum(0)
        else:
            self._momentum_batchcounter = 10
            self._trainmodel(0)
        return stepcost, avg_gradnorm

    def compute_validcost(self):
        n_batches = self._valid_inputs.shape[0] / self.batchsize
        validcost = 0.0
        for batch_idx in range(n_batches):
            validcost += self._compute_validcost(batch_idx)
        validcost /= n_batches
        return validcost

    def get_avg_gradnorm(self):
        avg_gradnorm = 0.0
        print self.gradnorm()
        for batch_idx in range(self._numbatches):
            self._batch_idx.set_value(batch_idx)
            tmp = self.gradnorm()
            avg_gradnorm += tmp / self._numbatches
        print avg_gradnorm
        return avg_gradnorm

    def step(self):
        cost = 0.0
        stepcount = 0.0

        self._epochcount += 1

        patience_expired = False
        for load_index in range(self._numloads):
            indices = np.random.permutation(self._loadsize)
            if self._inputs_type == 'h5':
                self._inputs_theano.set_value(
                    self._inputs.read(
                        start=load_index * self._loadsize,
                        stop=(load_index + 1) * self._loadsize)[indices])
            elif self._inputs_type == 'function':
                # if load has been used n times, gen new load
                if self._epochcount % self.numepochs_per_load == 0:
                    print 'using data function to generate new load...'
                    inp = np.empty((self._loadsize, ) + (self._inputs_fn().shape),
                                dtype=np.float32)
                    for i in xrange(self._loadsize):
                        inp[i] = self._inputs_fn()
                        if (i + 1) % 100 == 0:
                            print '{0}/{1}'.format(i + 1, self.loadsize)

                    self._inputs_theano.set_value(inp)
                    print 'done'
            else:
                self._inputs_theano.set_value(
                    self._inputs[load_index * self._loadsize + indices])
            for batch_index in self._rng.permutation(self._numbatches):
                stepcount += 1.0
                self._total_stepcount += 1.0
                stepcost, avg_gradnorm = self._trainsubstep(batch_index)
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)* \
                    stepcost
                if self._valid_inputs is not None:
                    if int(self._total_stepcount) % self._validation_frequency == 0:
                        print 'validation...'
                        #validate TODO: implement theano validation function
                        valid_cost = self.compute_validcost()
                        print 'valid_cost: {0}'.format(valid_cost)
                        if valid_cost < self._best_valid_cost * self._improvement_threshold:
                            self._patience = max(self._patience, self._total_stepcount * self._patience_increase)
                            for p in self._model.params:
                                self._best_valid_params[p.name] = p.get_value()
                            self._best_valid_cost = valid_cost
                            print 'new best valid cost: {0}'.format(self._best_valid_cost)
                            self.epoch_of_last_valid_improvement = self._epochcount
                    if self._patience <= self._total_stepcount:
                        #signal stop (returned at end of step)
                        patience_expired = True

            if self._verbose:
                print '> epoch {0:d}, load {1:d}/{2:d}, cost: {3:f}, avg. gradnorm: {4}'.format(
                    self._epochcount, load_index + 1, self._numloads, cost, avg_gradnorm)
            if np.isnan(cost):
                raise ValueError, 'Cost function returned nan!'
            elif np.isinf(cost):
                raise ValueError, 'Cost function returned infinity!'
        if patience_expired:
            print 'patience expired'
        return patience_expired


if __name__ == '__main__':
    import factoredGAE
    DATAPATH = os.environ['DATAPATH']
    datafile = tables.openFile(
        os.path.join(DATAPATH, 'Berkshft_1260_13_10.h5'))
    data = datafile.root.data_white[:10000, :]
    horizon = 10
    nvis = data.shape[-1] // horizon

    model = factoredGAE.FactoredGAE(
        numvisX=nvis, numvisY=nvis, numfac=256, nummap=128,
        numhid=0, output_type='real')

    trainer = SGD_Trainer(
        model, inputs=data[:, :2*nvis], batchsize=100,
        loadsize=10000, learningrate=0.0001)
    trainer.step()
    trainer.save('/tmp/test.pkl')
    trainer.load('/tmp/test.pkl')
    trainer.save('/tmp/test.h5')
