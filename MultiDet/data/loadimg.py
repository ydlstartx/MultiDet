from matplotlib import colors as mcolors
import matplotlib
import numpy as np
import random
from itertools import zip_longest
import mxnet as mx
from .genRecordIO import get_clsname

colors = list(mcolors.cnames.values())


def getRecTex(bbox, label, bcolor='r', tcolor='w'):
    rec = matplotlib.patches.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=bcolor, linewidth=3)
    
    if label is not None:
        tex = matplotlib.text.Text(bbox[0], bbox[1], label, color=tcolor,
                               verticalalignment='top', horizontalalignment='left',
                               fontsize=15, bbox=dict(facecolor=bcolor))
    else:
        tex = None
    return rec, tex


def showBBox(axes, img, bboxes, labels=None):
    axes.imshow(img)
    h, w, c = img.shape
    if labels is None:
        labels = []
    if bboxes.ndim == 1:
        bboxes = [bboxes]
    for i, (bbox, label) in enumerate(zip_longest(bboxes, labels, fillvalue=None)):
        if label != -1:
            if label is not None:
                text = get_clsname(int(label))
                bcolor = colors[int(label)]
                tcolor = colors[int(label) + 100]
            else:
                text = ''
                bcolor = colors[i%100]
                tcolor = 'w'
 
            RecPatch, TexArt = getRecTex(bbox, text, bcolor=bcolor,
                                         tcolor=tcolor)
            axes.add_patch(RecPatch)
            axes.add_artist(TexArt)


default_auglist = mx.image.CreateDetAugmenter((3,480,480),
                                              rand_crop=0.5, mean=None, std=None,
                                              min_object_covered=0.95)

auglist = mx.image.CreateDetAugmenter((3,480,480),
                                      rand_crop=0.5, mean=True, std=True,
                                      min_object_covered=0.95)

class DataIter:
    def __init__(self, idx, rec, batch_size, shuffle=False, aug_list=None, ctx=None):
        self.recordio = mx.recordio.MXIndexedRecordIO(idx, rec, 'r')
        self.indexes = self.recordio.keys
        self.num_img = len(self.indexes)
        self.batch_size = batch_size
        self.aug_list = aug_list
        self.ctx = ctx
        self.shuffle = shuffle

        self.rand_indexes = self.indexes.copy()
        self.reset()

    def _sample_from_ind(self, ind, aug_list=None):
        record = self.recordio.read_idx(ind)
        header, img = mx.recordio.unpack_img(record)
        img = img[:, :, ::-1]

        headlen = int(header.label[0])  
        labellen = int(header.label[1]) 
        numbox = int(header.label[2])

        label = header.label[headlen:].reshape((-1, labellen))
        img = mx.nd.array(img, ctx=self.ctx)

        if aug_list is not None:
            olabel = np.full(label.shape, -1, dtype='float32')
            labeltmp = label[:numbox]
            for aug in aug_list:
                img, labeltmp = aug(img, labeltmp)

            olabel[:labeltmp.shape[0]] = labeltmp
            label = olabel

        label = mx.nd.array(label, ctx=self.ctx)
        return img.transpose((2, 0, 1)), label

    def next_sample(self, aug_list=None):
        if self.flag:
            self.reset()
            raise StopIteration

        img, label = self._sample_from_ind(self.rand_indexes[self.cur],
                                           aug_list=aug_list)
        self.cur += 1
        if self.cur >= self.num_img:
            self.flag = 1

        return img, label

    def next_batch(self):
        labels = []
        imgs = []

        i = 0
        while i < self.batch_size:
            img, label = self.next_sample(aug_list=self.aug_list)
            imgs.append(img)
            labels.append(label)
            i += 1
            if self.cur >= self.num_img:
                break

        return mx.nd.stack(*imgs, axis=0), mx.nd.stack(*labels, axis=0)

    def tell(self):
        return self.cur

    def reset(self):
        self.cur = 0
        self.flag = 0
        if self.shuffle:
            random.shuffle(self.rand_indexes)

    def __next__(self):
        return self.next_batch()

    def __iter__(self):
        return self