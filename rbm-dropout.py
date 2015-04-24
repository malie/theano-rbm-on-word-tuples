import numpy
import theano
import theano.tensor as T
numpy.set_printoptions(linewidth=230, threshold=10000, edgeitems=18)

from codec import Codec
from visualize_weights import VisualizeWeights

class RBM:
    def __init__(self, num_visible, num_hidden, minibatch_size,
                 venabledp=1, henabledp=1):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.minibatch_size = minibatch_size;
        self.venabledp = venabledp
        self.henabledp = henabledp
        initial_weights = numpy.zeros((num_visible, num_hidden),
                                      dtype='float32')
        initial_hbias = numpy.zeros(num_hidden,
                                    dtype='float32')
        initial_vbias = numpy.zeros(num_visible,
                                    dtype='float32') - 3
        self.weights = theano.shared(value=initial_weights,
                                     name='weights')
        self.vbias = theano.shared(value=initial_vbias,
                                   name='vbias')
        self.hbias = theano.shared(value=initial_hbias,
                                   name='hbias')
        numpy_rng = numpy.random.RandomState(17)
        self.theano_rng = T.shared_randomstreams.RandomStreams(
            numpy_rng.randint(2**30))
            
        
    def propup(self, vis):
        '''Determine the activation probabilities of
        the hiddens given the states of the visibles.'''
        return T.nnet.sigmoid(T.dot(vis, self.weights)
                              / self.venabledp
                              + self.hbias)

    def sample_h_given_v(self, vis, henabled):
        '''Sample the state of the hiddens given the
        states of the visibles.'''
        h_probs = self.propup(vis)
        return (self.theano_rng.binomial(size=h_probs.shape,
                                         n=1,
                                         p=h_probs,
                                         dtype='float32')
                * henabled)

    def propdown(self, hid):
        '''Determine the activation probabilities of
        the visibles given the states of the hiddens.'''
        return T.nnet.sigmoid(T.dot(hid, self.weights.T)
                              / self.henabledp
                              + self.vbias)

    def sample_v_given_h(self, hid, venabled):
        '''Sample the state of the visibles given the
        states of the hidden.'''
        v_probs = self.propdown(hid)
        return (self.theano_rng.binomial(size=v_probs.shape,
                                         n=1,
                                         p=v_probs,
                                         dtype='float32')
                * venabled)

    def contrastive_divergence_1(self, v1, venabled, henabled):
        '''Determine the weight updates according to CD-1'''
        v1do = v1 * venabled
        h1 = self.sample_h_given_v(v1, henabled)
        v2 = self.sample_v_given_h(h1, venabled)
        h2p = self.propup(v2) * henabled
        updates = T.tensordot(v1, h1, [[0],[0]]) - T.tensordot(v2, h2p, [[0],[0]])
        f = 1.0 / self.minibatch_size
        return (updates * f,
                T.sum(v1 - v2, axis=0) * f,
                T.sum(h1 - h2p, axis=0) * f)

    def dropout_enabled_units(self):
        venabled = self.theano_rng.binomial(size=self.vbias.shape,
                                            n=1,
                                            p=self.venabledp,
                                            dtype='float32')
        henabled = self.theano_rng.binomial(size=self.hbias.shape,
                                            n=1,
                                            p=self.henabledp,
                                            dtype='float32')
        return (venabled, henabled)
        

    def cd1_fun(self, vis, learning_rate=0.5):
        (venabled, henabled) = self.dropout_enabled_units()
        (W, V, H) = self.contrastive_divergence_1(vis, venabled, henabled)
        return theano.function(
            [vis],
            (V, venabled, henabled),
            updates=[(self.weights,
                      T.cast(self.weights + W*learning_rate,
                             'float32')),
                     (self.vbias,
                      T.cast(self.vbias + V*learning_rate,
                             'float32')),
                     (self.hbias,
                      T.cast(self.hbias + H*learning_rate,
                             'float32'))])



def tuples_to_matrix(tuplesize, tuples, num_words):
    res = numpy.zeros((len(tuples), tuplesize*num_words),
                      dtype=numpy.float32)
    for i in range(len(tuples)):
        tup = tuples[i]
        for t in range(len(tup)):
            word = tup[t]
            res[i, word + num_words*t] = 1.0
    return res

        
def test():
    minibatch_size = 100
    num_words = 40
    tuplesize = 5
    num_visible = tuplesize*num_words
    num_hidden = 140

    codec = Codec(tuplesize, num_words)
    tuples = codec.tuples
    words = codec.words
    encoded = codec.tuples_to_matrix()
    (num_data, _) = encoded.shape

    print(words)
    print('data count: ', num_data)

    rbm = RBM(num_visible = num_visible,
              num_hidden = num_hidden,
              minibatch_size = minibatch_size,
              venabledp=1.0,
              henabledp=0.7)
    id_indices = numpy.random.randint(low=0, high=num_data, size=minibatch_size)
    input_data = T.constant(encoded[id_indices])

    #print(input_data)
    
    #print(rbm.propup(input_data).eval())

    #h1samples = rbm.sample_h_given_v(input_data).eval()
    #print(h1samples)

    #print(rbm.propdown(h1samples).eval())

    #v2samples = rbm.sample_v_given_h(h1samples).eval()
    #print(v2samples)

    #(W,H,V) = rbm.contrastive_divergence_1(input_data)
    #print(W.eval())
    #print(H.eval())
    #print(V.eval())


    all_h_enabled = numpy.ones(num_hidden)
    all_v_enabled = numpy.ones(num_visible)

    xvis = T.fmatrix('xvis')
    h1samples = rbm.sample_h_given_v(xvis, all_h_enabled)
    v2samples = rbm.sample_v_given_h(h1samples, all_v_enabled)
    sample_vhv = theano.function([xvis], v2samples)

    example_indices = numpy.random.randint(low=0, high=num_data, size=minibatch_size)
    example_input_data = encoded[example_indices]
    num_examples = min(10, minibatch_size)
    def show_examples():
        rec = sample_vhv(example_input_data)
        for example in range(num_examples):
            print('input words:',
                  [(t+1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if example_input_data[example, t*num_words + idx]])
            print('reconstructed words:',
                  [(t+1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if rec[example, t*num_words + idx]])

    def report_hidden():
        weights = rbm.weights.get_value()
        for h in range(num_hidden):
            print('hidden ', h)
            for block in range(tuplesize):
                for word in range(num_words):
                    w = weights[block*num_words+word, h]
                    if w > 0.5:
                        print('   %2i %8s  %4.1f' % (block, words[word], w))
        
    vis = T.fmatrix('vis')
    train = rbm.cd1_fun(vis)

    draw = VisualizeWeights('Dropout (vp:%4.2f, hp:%4.2f)' % (rbm.venabledp, rbm.henabledp),
                            rbm, tuplesize, words, num_hidden,
                            num_visible)
    for epoch in range(1000):
        show_examples()
        all_vdiffs = numpy.zeros(num_visible)
        print('epoch ', epoch)
        numpy.random.shuffle(encoded)
        for minibatch in range(num_data // minibatch_size):
            mb_start = minibatch * minibatch_size;
            mb_end = mb_start + minibatch_size;
            input_data_indices = numpy.arange(mb_start, mb_end)
            encoded_input = encoded[input_data_indices]
            input_data = encoded_input
            (vdiffs, venabled, henabled) = train(input_data)
            all_vdiffs = all_vdiffs + numpy.abs(vdiffs)
            #print('venabled', venabled)
            #print('henabled', henabled)
        print('reconstruction error: ', numpy.sum(all_vdiffs) * minibatch_size)
        #print(numpy.ndarray.astype(rbm.weights.get_value()*100, numpy.int32))
        #print(numpy.ndarray.astype(rbm.vbias.get_value()*100, numpy.int32))
        #print(numpy.ndarray.astype(rbm.hbias.get_value()*100, numpy.int32))
        draw.epoch_finished(epoch)
        report_hidden()

test()



