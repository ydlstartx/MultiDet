from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd


vgg16_config ={'conv':[(2, (64, 3, 1, 1)), 
                       (2, (128, 3, 1, 1)), 
                       (3, (256, 3, 1, 1)), 
                       (3, (512, 3, 1, 1)), 
                       (3, (512, 3, 1, 1))],
               'pool':[(2,2), (2,2), (2,2), (2,2), (2,2)]} 


class VGGBase(nn.Block):
    def __init__(self, config, use_batch=False, ceil_mode=True, 
                 init=mx.init.Xavier, **kwargs):
        super(VGGBase, self).__init__(**kwargs)

        self._config = config
        self._use_batch = use_batch
        self._ceil_mode = ceil_mode
        self._base_conv = 'base_conv_blk'
        self._base_pool = 'base_pool'
        self._init = init
        
        for i, conv in enumerate(self._config['conv'], 1):
            setattr(self, self._base_conv+'%d'%i, 
                    self._get_conv_blk(conv, self._use_batch, self._base_conv+'%d'%i))
        for i, pool in enumerate(self._config['pool'], 1):
            setattr(self, self._base_pool+'%d'%i,
                    self._get_pool(pool, self._ceil_mode, self._base_pool+'%d'%i))
    
    def _get_conv(self, c, k, s, p, d=1, prefix=None):
        conv = nn.Sequential(prefix=prefix)
        conv.add(nn.Conv2D(channels=c, kernel_size=k, strides=s, padding=p, 
                           dilation=d, weight_initializer=self._init()))
        if self._use_batch:
            conv.add(nn.BatchNorm())
        conv.add(nn.Activation('relu'))
        return conv
    
    def _get_conv_blk(self, blk_config, use_batch, prefix=None):
        blk = nn.Sequential(prefix=prefix)
        with blk.name_scope():
            if blk_config[0] > 0:
                (c, k, s, p) = blk_config[1]
                for i in range(1, blk_config[0]+1):
                    blk.add(self._get_conv(c, k, s, p, prefix='conv%d'%i))
            else:
                for i, (c, k, s, p) in enumerate(blk_config[1:], 1):
                    blk.add(self._get_conv(c, k, s, p, prefix='conv%d'%i))
        return blk
    
    def _get_pool(self, pool_config, ceil_mode, prefix=None):
        return nn.MaxPool2D(pool_size=pool_config[0], strides=pool_config[1],
                            ceil_mode=ceil_mode, prefix=prefix)
    
    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, self._base_conv+'%d'%i)(x)
            x = getattr(self, self._base_pool+'%d'%i)(x)
        return x

feature_config = {'conv':[(0, (256, 1, 1, 0), (512, 3, 2, 1)),
                          (0, (128, 1, 1, 0), (256, 3, 2, 1)),
                          (0, (128, 1, 1, 0), (256, 3, 1, 0)),
                          (0, (128, 1, 1, 0), (256, 3, 1, 0))]}


class VggFeature(VGGBase):
    def __init__(self, base_config, feature_config, use_batch=False, 
                 ceil_mode=True, init=mx.init.Xavier, **kwargs):
        super(VggFeature, self).__init__(base_config, use_batch=use_batch, 
                                         ceil_mode=ceil_mode, init=init,
                                         **kwargs)
        setattr(self, self._base_pool+'5', 
                nn.MaxPool2D(pool_size=3, strides=1, padding=1,
                            ceil_mode=self._ceil_mode, prefix=self._base_pool+'5'))
        
        self._feature_conv = 'feature_conv_blk'
        self._feature_config = feature_config
        
        self._conv4_scale = self.params.get('conv4_scale', shape=(1, 512, 1, 1),
                                            init=mx.init.Constant(20))
        
        self.feature_conv_blk6 = self._get_conv(1024, 3, 1, 6, 6)
        self.feature_conv_blk7 = self._get_conv(1024, 1, 1, 0, 1)
        
        for i, conv in enumerate(self._feature_config['conv'], 8):
            setattr(self, self._feature_conv+'%d'%i, 
                    self._get_conv_blk(conv, self._use_batch, self._feature_conv+'%d'%i))
            
    def forward(self, x):
        features = []
        for i in range(1, 4):
            x = getattr(self, self._base_conv + str(i))(x)
            x = getattr(self, self._base_pool + str(i))(x)

        x = getattr(self, self._base_conv+str(4))(x)
        x = nd.L2Normalization(x, mode='channel') * self._conv4_scale.data()
        features.append(x)
        x = getattr(self, self._base_pool + str(4))(x)
        
        x = getattr(self, self._base_conv+str(5))(x)
        x = getattr(self, self._base_pool + str(5))(x)
        
        x = self.feature_conv_blk6(x)
        x = self.feature_conv_blk7(x)
        features.append(x)
        
        for i in range(8, 12):
            x = getattr(self, self._feature_conv+str(i))(x)
            features.append(x)
        
        return features

def get_VGGFeature():
    return VggFeature(vgg16_config, feature_config)