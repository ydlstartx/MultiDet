import time
from mxnet import nd

class ctime():
    def __init__(self, prefix=''):
        self.prefix = prefix
    def __enter__(self, prefix=''):
        self.begin = time.time()
    def __exit__(self, *args):
        print(self.prefix, ': %.4f\n'%(time.time()-self.begin))



def getwh(scales, ratios, fw, fh, srmode):
    if srmode == 'few':
        num = scales.size + ratios.size - 1
        width = nd.zeros((num,))
        height = nd.zeros((num,))
        
        sqt_ratios = nd.sqrt(ratios)
        width[:ratios.size] = scales[0] * sqt_ratios
        height[:ratios.size] = width[:ratios.size] / ratios
        
        width[ratios.size:] = scales[1:] * sqt_ratios[0]
        height[ratios.size:] = width[ratios.size:] / ratios[0]
    else:
        rscales = nd.repeat(scales, ratios.size)
        rratios = nd.tile(ratios, scales.size)
        
        width = rscales * nd.sqrt(rratios)
        height = width / rratios
        
    width = width * fw
    height = height * fh
    
    return width, height


def getDefaultBoxes(fmap, s=None, r=None, 
                    offset=None, norm=None, clip=False, 
                    srmode='few', omode='flatten'):
    assert omode in ('flatten', 'stack')
    assert srmode in ('few', 'many')
    n, c, fh, fw = fmap.shape
    
    if s is None:
        scales = nd.array([1.])
    else:
        scales = nd.array(s)

    if r is None:
        ratios = nd.array([1.])
    else:
        ratios = nd.array(r)
        
    width, height = getwh(scales, ratios, fw, fh, srmode)
    
    nbox_per_pixel = width.size
    xcenter = nd.repeat(nd.arange(fw).reshape((1,-1)), fh, axis=0)
    ycenter = nd.repeat(nd.arange(fh).reshape((-1,1)), fw, axis=1)
    xycenters = nd.stack(xcenter, ycenter, axis=2)
    xycenters = nd.tile(xycenters, [1, 1, nbox_per_pixel*2])
    

    lu_rd_offset = nd.stack(width*-0.5, height*-0.5, width*0.5, height*0.5, axis=1)

    lu_rd_offset = lu_rd_offset.reshape((-1,))
    
    lu_rd_points = (xycenters + lu_rd_offset).reshape((fh, fw, nbox_per_pixel, 2, 2))
    
    if offset is None:
        offset = nd.array([0.5, 0.5])
    else:
        offset = nd.array(offset)
    assert offset.size <= 2
    
    if norm is None:
        norm = nd.array([fw, fh])
    else:
        norm = nd.array(norm)
    assert norm.size <= 2
    
    lu_rd_points = (lu_rd_points + offset) / norm
    
    if clip:
        nd.clip(lu_rd_points, a_min=0., a_max=1., out=lu_rd_points)
    
    if omode == 'flatten':
        lu_rd_points = lu_rd_points.reshape((1, -1, 4))
    else:
        lu_rd_points = lu_rd_points.reshape((1, fh, fw, nbox_per_pixel, 4))
    
    return lu_rd_points


def calIOU(anchor, gt):
    assert len(anchor.shape) in (1,2,3)
    assert len(gt.shape) in (1,2,3)
    
    anchor = anchor.reshape((-1,4))
    if len(gt.shape) < 3:
        gt = gt.reshape((1,1,4)) if len(gt.shape) == 1 else nd.expand_dims(gt, axis=0)
    anchor = nd.expand_dims(anchor, axis=1)
    gt = nd.expand_dims(gt, axis=1)
    
    max_tl = nd.maximum(nd.take(anchor, nd.array([0,1]), axis=-1), nd.take(gt, nd.array([0,1]), axis=-1))
    min_br = nd.minimum(nd.take(anchor, nd.array([2,3]), axis=-1), nd.take(gt, nd.array([2,3]), axis=-1))
    
    area = nd.prod(min_br-max_tl, axis=-1)
    i = nd.where((max_tl >= min_br).sum(axis=-1), nd.zeros_like(area), area)
    
    anchor_area = nd.prod(anchor[:,:,2:]-anchor[:,:,:2], axis=-1)
    gt_area = nd.prod(gt[:,:,:,2:]-gt[:,:,:,:2], axis=-1)
    total_area = anchor_area + gt_area - i
    iou = i / total_area
    
    return iou


def getUniqueMatch(iou, min_threshold=1e-12):
    N, M = iou.shape
    iouf = iou.reshape((-1,))
    
    argmax = nd.argsort(iouf, is_ascend=False)
    argrow = nd.floor(nd.divide(argmax, M))
    argcol = nd.modulo(argmax, M)

    uniquel = set()
    uniquer = set()
    match = nd.ones((N,)) * -1
    i = 0
    while True:
        if argcol[i].asscalar() not in uniquel and argrow[i].asscalar() not in uniquer:
            uniquel.add(argcol[i].asscalar())
            uniquer.add(argrow[i].asscalar())
            if iou[argrow[i], argcol[i]] > min_threshold:
                match[argrow[i]] = argcol[i]
        if len(uniquel) == M or len(uniquer) == N:
            break
        i += 1
    return match.reshape((1,-1))


def match(iou, threshould=0.5, share_max=False):
    B, N, M = iou.shape
    
    if share_max:
        result = nd.argmax(iou, axis=-1)
        result = nd.where(nd.max(iou, axis=-1) > threshould, result, nd.ones_like(result)*-1)
    else:
        match = [getUniqueMatch(i) for i in iou]
        result = nd.concat(*match, dim=0)
        argmax_row = nd.argmax(iou, axis=-1)
        max_row = nd.max(iou, axis=-1)
        argmax_row = nd.where(max_row > threshould, argmax_row, nd.ones_like(argmax_row)*-1)
        result = nd.where(result > -0.5, result, argmax_row)
        
    return result


def sample(match, cls_pred, iou, ratio=3, min_sample=0, threshold=0.5, do=True):
    if do is False:
        ones = nd.ones_like(match)
        sample = nd.where(match > -0.5, ones, ones*-1)
        return sample
    sample = nd.zeros_like(match)
    num_pos = nd.sum(match > -0.5, axis=-1)
    requre_neg = ratio * num_pos
    neg_mask = nd.where(match < -0.5, nd.max(iou, axis=-1) < threshold, sample)
    max_neg = neg_mask.sum(axis=-1)
    num_neg = nd.minimum(max_neg, nd.maximum(requre_neg, min_sample)).astype('int')
   
    neg_prob = cls_pred[:,:,0]
    max_value = nd.max(cls_pred, axis=-1, keepdims=True)
    score = max_value[:,:,0] - neg_prob + nd.log(
                                   nd.sum(
                                   nd.exp(cls_pred-max_value), axis=-1))

    score = nd.where(neg_mask, score, nd.zeros_like(score))
    argmax = nd.argsort(score, axis=-1, is_ascend=False)
    sample = nd.where(match > -0.5, nd.ones_like(sample), sample)
    
    for i, num in enumerate(num_neg):
        sample[i, argmax[i,:num.asscalar()]] = -1
    
    return sample


def label_box_cls(match, sample, gt_cls, ignore_label=-1):
    B, N = match.shape
    B, M = gt_cls.shape
    # (B,N,M)
    gt_cls = gt_cls.reshape((B,1,M))
    gt_cls = nd.broadcast_to(gt_cls, (B,N,M))
    # (B,N)
    label_cls = nd.pick(gt_cls, match, axis=-1) + 1
    label_cls = nd.where(sample > 0.5, label_cls, nd.ones_like(label_cls)*ignore_label)
    label_cls = nd.where(sample < -0.5, nd.zeros_like(label_cls), label_cls)
    # (B,N)
    label_mask = label_cls > -0.5
    return label_cls, label_mask


def corner_to_center(box, split=False, eps=1e-12):
    shape = box.shape
    assert len(shape) in (2,3) and shape[-1] == 4
    # (B,N,1) or (N,1)
    xmin, ymin, xmax, ymax = nd.split(box, 4, axis=-1)
    width = xmax - xmin
    height = ymax - ymin
    cx = xmin + width / 2
    cy = ymin + height / 2
    width = nd.where(width==0, nd.full(width.shape, eps), width)
    height = nd.where(height==0, nd.full(height.shape, eps), height)
    result = [cx, cy, width, height]
    if split:
        return result
    else:
        return nd.concat(*result, dim=-1)


def label_offset(anchors, bbox, match, sample, 
                 means=(0,0,0,0), stds=(0.1,0.1,0.2,0.2), flatten=True):
    anchors = anchors.reshape((-1,4))
    N, _ = anchors.shape
    B, M, _ = bbox.shape
    anchor_x, anchor_y, anchor_w, anchor_h = corner_to_center(anchors, split=True)
    
    bbox = bbox.reshape((B,1,M,4))
    bbox = nd.broadcast_to(bbox, (B,N,M,4))
    bbox = nd.stack(*[nd.pick(bbox[:,:,:,p], match) for p in range(4)], axis=-1)
    bbox_x, bbox_y, bbox_w, bbox_h = corner_to_center(bbox, split=True)
    
    offset_x = ((bbox_x - anchor_x) / anchor_w - means[0]) / stds[0]
    offset_y = ((bbox_y - anchor_y) / anchor_h - means[1]) / stds[1]
    offset_w = (nd.log(bbox_w/anchor_w) - means[2]) / stds[2]
    offset_h = (nd.log(bbox_h/anchor_h) - means[3]) / stds[3]
    offset = nd.concat(*(offset_x, offset_y, offset_w, offset_h), dim=-1)
    sample = sample.reshape((B,N,1))
    sample = nd.broadcast_to(sample, (B,N,4)) > 0.5
    
    anchor_offset = nd.where(sample, offset, nd.zeros_like(offset))
    anchor_mask = nd.where(sample, nd.ones_like(offset), nd.zeros_like(offset))
    
    if flatten:
        anchor_offset = anchor_offset.reshape((B,-1))
        anchor_mask = anchor_mask.reshape((B,-1))
        
    return anchor_mask, anchor_offset



def get_label(anchor, gt_label_offset, cls_pred, match_threshould=0.5):
    gt = gt_label_offset[:,:,1:]
    gt_cls = gt_label_offset[:,:,0]
    iou = calIOU(anchor, gt)
    mat = match(iou, threshould=match_threshould, share_max=False)
    samp = sample(mat, cls_pred, iou, ratio=3, min_sample=0, threshold=0.5, do=True)
    label_cls, label_mask = label_box_cls(mat, samp, gt_cls, ignore_label=-1)
    anchor_mask, anchor_offset = label_offset(anchor, gt, mat, samp)

    return label_cls, label_mask, anchor_offset, anchor_mask
