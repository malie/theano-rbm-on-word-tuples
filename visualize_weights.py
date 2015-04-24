import gizeh
import math
class VisualizeWeights:
    def __init__(self, title, rbm, tuplesize, words, num_hidden, num_visible=60):
        self.title = title
        self.rbm = rbm
        self.tuplesize = tuplesize
        self.words = words
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.width = (num_hidden+tuplesize*2)*22 + 200
        self.height = num_visible*22 + 300
    def epoch_finished(self, epoch):
        num_words = len(self.words)
        surface = gizeh.Surface(width=1200, height=self.height)
        background = gizeh.rectangle(lx=1200, ly=self.height, xy=[600,750], fill=(0,0,0))
        background.draw(surface)

        gizeh.text('%s   epoch %s' % (self.title, epoch),
                   fontfamily='Arial',
                   fontsize=20,
                   fill=(.6,.6,.6),
                   xy=(70,40),
                   h_align='left').draw(surface)
        
        weights = self.rbm.weights.get_value()
        for block in range(self.tuplesize):
            for word in range(num_words):
                x = 40
                y = 70+22*(block*(num_words+2)+word)
                gizeh.text('%s %s' % (block, self.words[word]),
                           fontfamily='Arial',
                           fontsize=12,
                           fill=(.7,.7,.7),
                           xy=(x,y),
                           h_align='left').draw(surface)
        for block in range(self.tuplesize):
            for word in range(num_words):
                for h in range(self.num_hidden):
                    vis = block*num_words+word
                    w = weights[vis, h]
                    r = math.log(1+abs(w))*4
                    x = 100+22*h
                    y = 70+22*(block*(num_words+2)+word)
                    col = (1,1,0)
                    if w < 0:
                        col = (.4,.4,.4)
                    circle = gizeh.circle(r=r, xy=[x,y], fill=col)
                    circle.draw(surface)
        surface.write_to_png("epoch-%03d.png" % epoch)


