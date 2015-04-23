from read import read_odyssey_tuples
import numpy

class Codec:
    def __init__(self, tuplesize, num_words):
        self.tuplesize = tuplesize
        self.num_words = num_words
        (self.tuples, self.words) = read_odyssey_tuples(tuplesize, num_words)
    def tuples_to_matrix(self):
        ntuples = len(self.tuples)
        num_visible = self.tuplesize*self.num_words
        res = numpy.zeros((ntuples, num_visible),
                          dtype=numpy.float32)
        for i in range(ntuples):
            tup = self.tuples[i]
            for t in range(len(tup)):
                word = tup[t]
                res[i, word + self.num_words*t] = 1.0
        return res
