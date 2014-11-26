import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class Pgp3layer(object):
    def __init__(self, numvis, numnote, numfac, numvel, numvelfac, numacc, numaccfac, numjerk, numframes_to_train, numframes_to_predict, output_type='real', coststart=4, vis_corruption_type="zeromask", vis_corruption_level=0.0, numpy_rng=None, theano_rng=None):
        self.numvis = numvis
        self.numnote = numnote
        self.numfac = numfac
        self.numvel = numvel
        self.numvelfac = numvelfac
        self.numacc = numacc
        self.numaccfac = numaccfac
        self.numjerk = numjerk
        self.numframes_to_train = numframes_to_train
        self.numframes_to_predict = numframes_to_predict
        self.output_type = output_type
        self.vis_corruption_type   = vis_corruption_type
        self.vis_corruption_level  = theano.shared(value=numpy.array([vis_corruption_level]), name='vis_corruption_level')
        self.autonomy = theano.shared(value=numpy.array([0.5]).astype("float32"), name='autonomy')
        self.coststart = coststart
        self.inputs = T.matrix(name='inputs') 

        if not numpy_rng:  
            self.numpy_rng = numpy.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        self.wxf_left = theano.shared(value = self.numpy_rng.normal(size=(numvis+numnote, numfac)).astype(theano.config.floatX)*0.01, name='wxf_left')
        self.wxf_right = theano.shared(value = self.numpy_rng.normal(size=(numvis+numnote, numfac)).astype(theano.config.floatX)*0.01, name='wxf_right')
        self.wv = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numfac, numvel)).astype(theano.config.floatX), name='wv')
        self.wvf_left = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvel, numvelfac)).astype(theano.config.floatX), name='wvf_left')
        self.wvf_right = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvel, numvelfac)).astype(theano.config.floatX), name='wvf_right')
        self.wa = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvelfac, numacc)).astype(theano.config.floatX), name='wa')
        self.waf_left = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numacc, numaccfac)).astype(theano.config.floatX), name='waf_left')
        self.waf_right = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numacc, numaccfac)).astype(theano.config.floatX), name='waf_right')
        self.wj = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numaccfac, numjerk)).astype(theano.config.floatX), name='wj')
        self.bx = theano.shared(value = 0.0*numpy.ones(numvis+numnote, dtype=theano.config.floatX), name='bx')
        self.bv = theano.shared(value = 0.0*numpy.ones(numvel, dtype=theano.config.floatX), name='bv')
        self.ba = theano.shared(value = 0.0*numpy.ones(numacc, dtype=theano.config.floatX), name='ba')
        self.bj = theano.shared(value = 0.0*numpy.ones(numjerk, dtype=theano.config.floatX), name='bj')
        self.params = [self.wxf_left, self.wxf_right, self.wv, self.wvf_left, self.wvf_right, self.wa, self.waf_left, self.waf_right, self.wj, self.bx, self.bv, self.ba, self.bj, self.autonomy]

        self._inputframes = [None] * self.numframes_to_predict
        self._inputframes_and_notebook = [None] * self.numframes_to_predict
        self._xfactors_left = [None] * self.numframes_to_predict
        self._xfactors_right = [None] * self.numframes_to_predict
        self._vels = [None] * self.numframes_to_predict
        self._accs = [None] * self.numframes_to_predict
        self._prejerks = [None] * self.numframes_to_predict
        self._recons_with_notebook = [None] * self.numframes_to_predict

        #extract all input frames and project onto input/output filters: 
        for t in range(self.numframes_to_predict):
            if t < self.numframes_to_train:
                self._inputframes[t] = self.inputs[:, t*numvis:(t+1)*numvis]
            else:
                self._inputframes[t] = T.zeros((self._inputframes[0].shape[0], self.numvis)) 

            if t>3:
                if self.vis_corruption_type=='zeromask':
                    self._inputframes[t] = theano_rng.binomial(size=self._inputframes[t].shape, n=1, p=1.0-self.vis_corruption_level, dtype=theano.config.floatX) * self._inputframes[t]
                elif self.vis_corruption_type=='mixedmask':
                    self._inputframes[t] = theano_rng.binomial(size=self._inputframes[t].shape, n=1, p=1.0-self.vis_corruption_level/2, dtype=theano.config.floatX) * self._inputframes[t]
                    self._inputframes[t] = (1-theano_rng.binomial(size=self._inputframes[t].shape, n=1, p=1.0-self.vis_corruption_level/2, dtype=theano.config.floatX)) * self._inputframes[t]
                elif self.vis_corruption_type=='gaussian':
                    self._inputframes[t] = theano_rng.normal(size=self._inputframes[t].shape, avg=0.0, std=self.vis_corruption_level, dtype=theano.config.floatX) + self._inputframes[t]
                else:
                    assert False, "vis_corruption type not understood"
            self._inputframes_and_notebook[t] = T.concatenate((self._inputframes[t], T.zeros((self._inputframes[t].shape[0], self.numnote))),1)
            self._recons_with_notebook[t] = self._inputframes_and_notebook[t]

        for t in range(4, self.numframes_to_predict):
            self._xfactors_left[t-4] = T.dot(self._recons_with_notebook[t-4], self.wxf_left)
            self._xfactors_right[t-4] = T.dot(self._recons_with_notebook[t-4], self.wxf_right)
            self._xfactors_left[t-3] = T.dot(self._recons_with_notebook[t-3], self.wxf_left)
            self._xfactors_right[t-3] = T.dot(self._recons_with_notebook[t-3], self.wxf_right)
            self._xfactors_left[t-2] = T.dot(self._recons_with_notebook[t-2], self.wxf_left)
            self._xfactors_right[t-2] = T.dot(self._recons_with_notebook[t-2], self.wxf_right)
            self._xfactors_left[t-1] = T.dot(self._recons_with_notebook[t-1], self.wxf_left)
            self._xfactors_right[t-1] = T.dot(self._recons_with_notebook[t-1], self.wxf_right)
            self._xfactors_left[t] = T.dot(self._recons_with_notebook[t], self.wxf_left)
            self._xfactors_right[t] = T.dot(self._recons_with_notebook[t], self.wxf_right)

            #re-infer current velocities v12 and v23: 
            self._prevel01 = T.dot(self._xfactors_left[t-4]*self._xfactors_right[t-3], self.wv)+self.bv
            self._prevel12 = T.dot(self._xfactors_left[t-3]*self._xfactors_right[t-2], self.wv)+self.bv
            self._prevel23 = T.dot(self._xfactors_left[t-2]*self._xfactors_right[t-1], self.wv)+self.bv
            self._prevel34 = T.dot(self._xfactors_left[t-1]*self._xfactors_right[t  ], self.wv)+self.bv

            #re-infer acceleration a123: 
            self._preacc012 = T.dot(T.dot(T.nnet.sigmoid(self._prevel01), self.wvf_left)*T.dot(T.nnet.sigmoid(self._prevel12), self.wvf_right), self.wa)+self.ba
            self._preacc123 = T.dot(T.dot(T.nnet.sigmoid(self._prevel12), self.wvf_left)*T.dot(T.nnet.sigmoid(self._prevel23), self.wvf_right), self.wa)+self.ba
            self._preacc234 = T.dot(T.dot(T.nnet.sigmoid(self._prevel23), self.wvf_left)*T.dot(T.nnet.sigmoid(self._prevel34), self.wvf_right), self.wa)+self.ba

            if t==4:
                self._prejerks[t-1] = T.dot(T.dot(T.nnet.sigmoid(self._preacc012), self.waf_left)*T.dot(T.nnet.sigmoid(self._preacc123), self.waf_right), self.wj)+self.bj

            #infer jerk as weighted sum of past and re-infered: 
            self._prejerks[t] = T.nnet.sigmoid(self.autonomy[0])*self._prejerks[t-1]+(1-T.nnet.sigmoid(self.autonomy[0]))*(T.dot(T.dot(T.nnet.sigmoid(self._preacc123), self.waf_left)*T.dot(T.nnet.sigmoid(self._preacc234), self.waf_right), self.wj)+self.bj)

            #fill in all remaining activations from top-level jerk and past: 
            self._accs[t] = T.nnet.sigmoid(T.nnet.sigmoid(self.autonomy[0])*(T.dot(T.dot(T.nnet.sigmoid(self._prejerks[t]), self.wj.T) * T.dot(self._preacc123, self.waf_left), self.waf_right.T) + self.ba) + (1.0-T.nnet.sigmoid(self.autonomy[0]))*self._preacc234)
            self._vels[t] = T.nnet.sigmoid(T.nnet.sigmoid(self.autonomy[0])*(T.dot(T.dot(self._accs[t], self.wa.T)*T.dot(self._prevel23,self.wvf_left), self.wvf_right.T)+self.bv) + (1.0-T.nnet.sigmoid(self.autonomy[0]))*self._prevel34)
            self._recons_with_notebook[t] = T.dot(T.dot(self._recons_with_notebook[t-1],self.wxf_left)*T.dot(self._vels[t], self.wv.T),self.wxf_right.T) + self.bx

        self._prediction = T.concatenate([pred[:,:self.numvis] for pred in self._recons_with_notebook], 1)
        self._notebook = T.concatenate([pred[:,self.numvis:] for pred in self._recons_with_notebook], 1)
        if self.output_type == 'binary':
            self._prediction_for_training = T.concatenate([T.nnet.sigmoid(pred[:,:self.numvis]) for pred in self._recons_with_notebook[self.coststart:self.numframes_to_train]], 1)
        else:
            self._prediction_for_training = T.concatenate([pred[:,:self.numvis] for pred in self._recons_with_notebook[self.coststart:self.numframes_to_train]], 1)

        if self.output_type == 'real':
            self._cost = T.mean((self._prediction_for_training - self.inputs[:,self.coststart*self.numvis:self.numframes_to_train*self.numvis])**2)
        elif self.output_type == 'binary':
            self._cost = -T.mean(self.inputs[:,self.coststart*self.numvis:self.numframes_to_train*self.numvis]*T.log(self._prediction_for_training) 
                                    + 
                                 (1.0-self.inputs[:,self.coststart*self.numvis:self.numframes_to_train*self.numvis])*T.log(1.0-self._prediction_for_training))

        self._grads = T.grad(self._cost, self.params)

        self.prediction = theano.function([self.inputs], self._prediction)
        self.notebook = theano.function([self.inputs], self._notebook)
        self.vels = [theano.function([self.inputs], v) for v in self._vels[4:]]
        self.accs = [theano.function([self.inputs], a) for a in self._accs[4:]]
        self.jerks = [theano.function([self.inputs], j) for j in self._prejerks[4:]]
        self.cost = theano.function([self.inputs], self._cost)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def predict(self, seedframes, numframes=10):
        numcases = seedframes.shape[0]
        frames_and_notes = [numpy.concatenate((seedframes[:,i*self.numvis:(i+1)*self.numvis], numpy.zeros((numcases, self.numnote), dtype="float32")),1) for i in range(seedframes.shape[1]/self.numvis)] 
        for i in range(seedframes.shape[1]/self.numvis, numframes):
            frames_and_notes.append(numpy.zeros((numcases,self.numnote+self.numvis),dtype="float32"))
        firstprejerk = theano.function([self._inputframes_and_notebook[0], self._inputframes_and_notebook[1],self._inputframes_and_notebook[2],self._inputframes_and_notebook[3]], self._prejerks[3])
        prejerk = firstprejerk(frames_and_notes[0], frames_and_notes[1], frames_and_notes[2], frames_and_notes[3])

        next_prediction_and_jerk = theano.function([self._inputframes_and_notebook[1],self._inputframes_and_notebook[2],self._inputframes_and_notebook[3], self._inputframes_and_notebook[4], self._prejerks[3]], T.concatenate((self._recons_with_notebook[4], self._prejerks[4]), 1))

        preds = numpy.concatenate((seedframes[:,:self.numvis*4], numpy.zeros((numcases,(numframes-4)*self.numvis),dtype="float32")), 1)
        for t in range(4, numframes):
            preds_notebook_jerks = next_prediction_and_jerk(frames_and_notes[t-3], frames_and_notes[t-2], frames_and_notes[t-1], frames_and_notes[t], prejerk)
            frames_and_notes[t][:,:] = preds_notebook_jerks[:,:self.numvis+self.numnote]
            prejerk = preds_notebook_jerks[:,self.numvis+self.numnote:]
            preds[:,t*self.numvis:(t+1)*self.numvis] = preds_notebook_jerks[:,:self.numvis]
        return preds

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value(borrow=False).flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

    def normalizefilters(self):
        wxf_left = self.wxf_left.get_value(borrow=True)
        wxf_left -= wxf_left.mean(0)[None,:]
        norms = numpy.sqrt((wxf_left**2).sum(0))[None,:]
        self.wxf_left.set_value(wxf_left*norms.mean()/norms,borrow=True)
        wxf_right = self.wxf_right.get_value(borrow=True)
        wxf_right -= wxf_right.mean(0)[None,:]
        norms = numpy.sqrt((wxf_right**2).sum(0))[None,:]
        self.wxf_right.set_value(wxf_right*norms.mean()/norms,borrow=True)


