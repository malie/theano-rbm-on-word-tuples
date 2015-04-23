import numpy
import theano
import theano.tensor as T
numpy.set_printoptions(linewidth=200, threshold=10000)

from codec import Codec
from visualize_weights import VisualizeWeights

class RBM1:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        initial_weights = numpy.zeros((num_visible, num_hidden),
                                      dtype='float32')
        initial_hbias = numpy.zeros(num_hidden,
                                    dtype='float32') - 0
        initial_vbias = numpy.zeros(num_visible,
                                    dtype='float32') - 2.3
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
                              + self.hbias)

    def sample_h_given_v(self, vis):
        '''Sample the state of the hiddens given the
        states of the visibles.'''
        h_probs = self.propup(vis)
        return self.theano_rng.binomial(size=h_probs.shape,
                                        n=1,
                                        p=h_probs,
                                        dtype='float32')

    def propdown(self, hid):
        '''Determine the activation probabilities of
        the visibles given the states of the hiddens.'''
        return T.nnet.sigmoid(T.dot(hid, self.weights.T)
                              + self.vbias)

    def sample_v_given_h(self, hid):
        '''Sample the state of the visibles given the
        states of the hidden.'''
        v_probs = self.propdown(hid)
        return self.theano_rng.binomial(size=v_probs.shape,
                                        n=1,
                                        p=v_probs,
                                        dtype='float32')

    def contrastive_divergence_1(self, v1):
        '''Determine the weight updates according to CD-1'''
        h1 = self.sample_h_given_v(v1)
        v2 = self.sample_v_given_h(h1)
        h2p = self.propup(v2)
        return (T.outer(v1, h1) - T.outer(v2, h2p),
                v1 - v2,
                h1 - h2p)

    def cd1_fun(self, vis, learning_rate=0.01):
        (W, V, H) = self.contrastive_divergence_1(vis)
        return theano.function(
            [vis],
            V,
            updates=[(self.weights,
                      T.cast(self.weights + W*learning_rate,
                             'float32')),
                     (self.vbias,
                      T.cast(self.vbias + V*learning_rate,
                             'float32')),
                     (self.hbias,
                      T.cast(self.hbias + H*learning_rate,
                             'float32'))])


def test():
    num_words = 20
    tuplesize = 3
    num_visible = tuplesize*num_words
    num_hidden = 30
    codec = Codec(tuplesize, num_words)
    tuples = codec.tuples
    words = codec.words
    encoded = codec.tuples_to_matrix()
    (num_data, _) = encoded.shape
    print('data count: ', num_data)

    rbm = RBM1(num_visible = num_visible,
               num_hidden = num_hidden)
    input_data = T.constant(encoded[2])
    #print(pairs[2])
    #print(encoded[2])

    #print(input_data)
    
    #print(rbm.propup(input_data).eval())

    h1samples = rbm.sample_h_given_v(input_data).eval()
    #print(h1samples)

    #print(rbm.propdown(h1samples).eval())

    v2samples = rbm.sample_v_given_h(h1samples).eval()
    #print(v2samples)

    (W,H,V) = rbm.contrastive_divergence_1(input_data)
    #print(W.eval())

    

    xvis = T.fvector('xvis')
    h1samples = rbm.sample_h_given_v(xvis)
    v2samples = rbm.sample_v_given_h(h1samples)
    sample_vhv = theano.function([xvis], v2samples)

    num_examples = 20
    example_indices = numpy.random.randint(low=0, high=num_data, size=num_examples)
    def show_examples():
        for example in example_indices:
            dat = encoded[example]
            v2samples = sample_vhv(dat)
            print('input words:',
                  [(t+1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if encoded[example, t*num_words + idx]])
            print('reconstructed words:',
                  [(t+1, words[idx])
                   for t in range(tuplesize)
                   for idx in range(num_words)
                   if v2samples[t*num_words + idx]])
            print('')
    def report_hidden():
        weights = rbm.weights.get_value()
        for h in range(num_hidden):
            print('hidden ', h)
            for block in range(tuplesize):
                for word in range(num_words):
                    w = weights[block*num_words+word, h]
                    if w > 0.5:
                        print('   %2i %8s  %4.1f' % (block, words[word], w))


    vis = T.fvector('vis')
    train = rbm.cd1_fun(vis)
    input_data = numpy.reshape(encoded[2],
                               num_visible)
    train(input_data)
    print(rbm.weights.get_value())

    draw = VisualizeWeights('', rbm, tuplesize, words, num_hidden)

    for epoch in range(500):
        show_examples()
        all_vdiffs = numpy.zeros(num_visible)
        print('epoch ', epoch)
        for i in range(num_data):
            input_data = numpy.reshape(encoded[i],
                                       num_visible)
            vdiffs = train(input_data)
            all_vdiffs = all_vdiffs + numpy.abs(vdiffs)
        print('reconstruction error: ', numpy.sum(all_vdiffs))
        print(T.cast(rbm.weights.get_value()*100, 'int32').eval())
        draw.epoch_finished(epoch)
        report_hidden()

test()

# turn images to an animated gif
# convert -delay 100 epoch-*.png epochs.gif


# show diffs per data item
# show number of hiddens activated, per data item
# negative bias for hiddens?
# what is each hidden doing?

# keep out a test set, that show's how the rbm generalizes

# init vbias better? from the terms in hinton practical paper?

