from .basenet import get_VGGFeature
from .box import getDefaultBoxes
from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx
import time


scales = [[0.1, 0.14], [0.2, 0.27], [0.37, 0.44], 
          [0.54, 0.62], [0.71, 0.79], [0.88, 0.96]]
ratios = [[1,2,0.5], [1,2,3,0.5,1/3], [1,2,3,0.5,1/3], 
          [1,2,3,0.5,1/3], [1,2,0.5], [1,2,0.5]]


class ctime:
    def __init__(self, prefix=''):
        self.prefix = prefix
    def __enter__(self):
        self.begin = time.time()
    def __exit__(self, *args):
        print(self.prefix, ' :%.3f'%(time.time()-self.begin))


class SSD(nn.Block):
    def __init__(self, base_net, scales, ratios, num_cls=20, 
                 init=mx.init.Xavier, **kwargs):
        super(SSD, self).__init__(**kwargs)
        
        assert len(scales) == len(ratios)
        self.base_net = base_net
        self.scales = scales
        self.ratios = ratios
        self.num_cls = num_cls
        self.init = init
        self.cls_pred = 'cls_pred'
        self.offset_pred = 'offset_pred'
        self.dbox = 'dbox'
        
        for i, (scale, ratio) in enumerate(zip(self.scales, self.ratios)):
            num_box = len(scale) + len(ratio) - 1
            setattr(self, self.cls_pred+str(i), 
                    self._get_pred(num_box, self.num_cls, mode='cls', prefix='cls%d'%i))
            setattr(self, self.offset_pred+str(i),
                    self._get_pred(num_box, mode='offset', prefix='offset%d'%i))
        

    def _get_pred(self, num_box, num_cls='', mode='cls', prefix=None):
        assert mode in ('cls', 'offset')
        if mode == 'cls':
            channels = (num_cls + 1) * num_box
        else:
            channels = 4 * num_box
        blk = nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, 
                        activation='relu', weight_initializer=self.init(), prefix=prefix)
        return blk
        
    def forward(self, x):
        features = self.base_net(x)
        assert len(features) == len(self.scales)
        
        cls_preds = []
        offset_preds = []
        dboxes = []
        for i, feature in enumerate(features):
            dbox = getattr(self, self.dbox+str(i), None)
            if dbox is None:
                with ctime('build dbox\n'):
                    print(feature.shape)
                    setattr(self, self.dbox+str(i),
                        getDefaultBoxes(feature, s=self.scales[i], r=self.ratios[i]))
                dbox = getattr(self, self.dbox+str(i))
                
            offset_pred = getattr(self, self.offset_pred+str(i))(feature)
            cls_pred = getattr(self, self.cls_pred+str(i))(feature)
            
            cls_preds.append(cls_pred.transpose((0,2,3,1)).flatten())
            offset_preds.append(offset_pred.transpose((0,2,3,1)).flatten())
            dboxes.append(dbox)
            
                
        return (nd.concat(*dboxes, dim=1), 
                nd.concat(*cls_preds, dim=1).reshape((0,-1,self.num_cls+1)),
                nd.concat(*offset_preds, dim=1))


def get_SSD300():
    base_net = get_VGGFeature()
    ssd = SSD(base_net, scales, ratios)
    return ssd