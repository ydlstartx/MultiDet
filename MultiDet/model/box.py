import numpy as np
from collections import abc
import numbers


def getnp(data):
    if data is None:
        data = np.array([1.], dtype='float')
    elif isinstance(data, numbers.Real):
        data = np.array([data], dtype='float')
    elif isinstance(data, abc.Sequence):
        data = np.array(data, dtype='float')
        assert len(data.shape) == 1
    return data


def getwh(scales, ratios, fw, fh, srmode):
    if srmode == 'few':
        num = scales.size + ratios.size - 1
        width = np.zeros((num,))
        height = np.zeros((num,))
        
        sqt_ratios = np.sqrt(ratios)
        width[:ratios.size] = scales[0] * sqt_ratios
        height[:ratios.size] = width[:ratios.size] / ratios
        
        width[ratios.size:] = scales[1:] * sqt_ratios[0]
        height[ratios.size:] = width[ratios.size:] / ratios[0]
    else:
        rscales = np.repeat(scales.reshape((-1,1)), ratios.size, axis=1).flatten()
        rratios = np.tile(ratios, scales.size)
        
        width = rscales * np.sqrt(rratios)
        height = width / rratios
        
    width = width * fw
    height = height * fh
    
    return width, height


def getDefaultBoxes(fmap, scales=None, ratios=None, 
                    offset=None, norm=None, clip=False, 
                    srmode='few', omode='flatten'):
    assert omode in ('flatten', 'stack')
    assert srmode in ('few', 'many')
    n, c, fh, fw = fmap.shape
    
    scales = scales if type(scales).__name__ == 'ndarray' else getnp(scales)
    ratios = ratios if type(ratios).__name__ == 'ndarray' else getnp(ratios)
        
    width, height = getwh(scales, ratios, fw, fh, srmode)
    
    nbox_per_pixel = width.size
    xcenter, ycenter = np.meshgrid(np.arange(fw), np.arange(fh))
    xycenters = np.stack((xcenter, ycenter), axis=2)
    xycenters = np.tile(xycenters, [1, 1, nbox_per_pixel*2])
    

    lu_rd_offset = np.stack((width, height, width, height), axis=1) *\
                   np.array([-1, -1 , 1, 1]) / 2
    lu_rd_offset = lu_rd_offset.flatten()
    
    lu_rd_points = (xycenters + lu_rd_offset).reshape((fh, fw, nbox_per_pixel, 2, 2))
    
    if offset is None:
        offset = np.array([0.5, 0.5])
    else:
        offset = np.array(offset)
    assert offset.size <= 2
    
    if norm is None:
        norm = np.array([fw, fh])
    else:
        norm = np.array(norm)
    assert norm.size <= 2
    
    lu_rd_points = (lu_rd_points + offset) / norm
    
    if clip:
        with ctime():
            np.clip(lu_rd_points, 0., 1., out=lu_rd_points)
    
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
        gt = gt.reshape((1,1,4)) if len(gt.shape) == 1 else np.expand_dims(gt, axis=0)
    anchor = np.expand_dims(anchor, axis=1)
    gt = np.expand_dims(gt, axis=1)
    
    max_tl = np.maximum(np.take(anchor, [0,1], axis=-1), np.take(gt, [0,1], axis=-1))
    min_br = np.minimum(np.take(anchor, [2,3], axis=-1), np.take(gt, [2,3], axis=-1))
    
    area = np.prod(min_br-max_tl, axis=-1)
    i = np.where((max_tl < min_br).all(axis=-1), area, np.zeros_like(area))
    
    anchor_area = np.prod(anchor[:,:,2:]-anchor[:,:,:2], axis=-1)
    gt_area = np.prod(gt[:,:,:,2:]-gt[:,:,:,:2], axis=-1)
    total_area = anchor_area + gt_area - i
    iou = i / total_area
    
    return iou


def getUniqueMatch(iou, min_threshold=1e-12):
    N, M = iou.shape
    iouf = iou.flatten()
    
    argmax = np.argsort(iouf)[::-1]
    argrow, argcol = np.divmod(argmax, M)

    uniquel = set()
    uniquer = set()
    match = np.ones((N,)) * -1
    i = 0
    while True:
        if argcol[i] not in uniquel and argrow[i] not in uniquer:
            uniquel.add(argcol[i])
            uniquer.add(argrow[i])
            if iou[argrow[i], argcol[i]] > min_threshold:
                match[argrow[i]] = argcol[i]
        if len(uniquel) == M or len(uniquer) == N:
            break
        i += 1
    return match.reshape((1,-1))


def match(iou, threshould=0.5, share_max=False):
    B, N, M = iou.shape
    
    if share_max:
        result = np.argmax(iou, axis=-1)
        result = np.where(np.max(iou, axis=-1) > threshould, result, np.ones_like(result)*-1)
    else:
        match = [getUniqueMatch(i) for i in iou]
        result = np.concatenate(match, axis=0)
        argmax_row = np.argmax(iou, axis=-1)
        max_row = np.max(iou, axis=-1)
        argmax_row = np.where(max_row > threshould, argmax_row, np.ones_like(argmax_row)*-1)
        result = np.where(result > -0.5, result, argmax_row)
        
    return result


def sample(match, cls_pred, iou, ratio=3, min_sample=0, threshold=0.5, do=True):
    if do is False:
        ones = np.ones_like(match)
        sample = np.where(match > -0.5, ones, ones*-1)
        return sample
    sample = np.zeros_like(match)
    num_pos = np.sum(match > -0.5, axis=-1)
    requre_neg = ratio * num_pos
    neg_mask = np.where(match < -0.5, np.max(iou, axis=-1) < threshold, sample)
    max_neg = neg_mask.sum(axis=-1)
    num_neg = np.minimum(max_neg, np.maximum(requre_neg, min_sample)).astype('int')
   
    neg_prob = np.take(cls_pred, 0, axis=-1)
    max_value = np.max(cls_pred, axis=-1)
    score = max_value - neg_prob + np.log(
                                   np.sum(
                                   np.exp(cls_pred-max_value[:,:,np.newaxis]), axis=-1))
    score = np.where(neg_mask, score, np.zeros_like(score))
    argmax = np.argsort(score, axis=-1)[:,::-1]
    sample[match>-0.5] = 1
    
    for i, num in enumerate(num_neg):
        sample[i, argmax[i,:num]] = -1
    
    return sample


def np_pick(data, pick_array, axis=-1): 
    pick_array = pick_array.astype('int')
    data_shape = list(data.shape) 
    pick_array_shape = list(pick_array.shape)
    data_shape.pop(axis)  
    assert data_shape == pick_array_shape
    
    grid = np.indices(data_shape) 
    grid = grid.tolist() 
    if axis != -1:
        grid.insert(axis, pick_array) 
    elif axis == -1 or axis == len(data_shape) + 1:
        grid.append(pick_array)
    grid = tuple(grid)
    return data[grid]


def label_box_cls(match, sample, gt_cls, ignore_label=-1):
    B, N = match.shape
    B, M = gt_cls.shape
    gt_cls = np.broadcast_to(gt_cls[:,np.newaxis,:], (B,N,M))
    label_cls = np_pick(gt_cls, match, axis=-1) + 1
    label_cls = np.where(sample > 0.5, label_cls, np.ones_like(label_cls)*ignore_label)
    label_cls = np.where(sample < -0.5, np.zeros_like(label_cls), label_cls)
    label_mask = (label_cls > -0.5).astype('int')
    return label_cls, label_mask


def corner_to_center(box, split=False):
    shape = box.shape
    assert len(shape) in (2,3) and shape[-1] == 4
    xmin, ymin, xmax, ymax = np.split(box, 4, axis=-1)
    width = xmax - xmin
    height = ymax - ymin
    cx = xmin + width / 2
    cy = ymin + height / 2
    result = [cx, cy, width, height]
    if split:
        return result
    else:
        return np.concatenate(result, axis=-1)


def label_offset(anchors, bbox, match, sample, 
                 means=(0,0,0,0), stds=(0.1,0.1,0.2,0.2), flatten=True):
    anchors = anchors.reshape((-1,4))
    N, _ = anchors.shape
    B, M, _ = bbox.shape
    anchor_x, anchor_y, anchor_w, anchor_h = corner_to_center(anchors, split=True)
    
    bbox = np.broadcast_to(bbox[:,np.newaxis,:,:], (B,N,M,4))
    bbox = np.stack([np_pick(np.take(bbox, p, axis=-1), match) for p in range(4)], axis=-1)
    bbox_x, bbox_y, bbox_w, bbox_h = corner_to_center(bbox, split=True)
    
    offset_x = ((bbox_x - anchor_x) / anchor_w - means[0]) / stds[0]
    offset_y = ((bbox_y - anchor_y) / anchor_h - means[1]) / stds[1]
    offset_w = (np.log(bbox_w/anchor_w) - means[2]) / stds[2]
    offset_h = (np.log(bbox_h/anchor_h) - means[3]) / stds[3]
    offset = np.concatenate((offset_x, offset_y, offset_w, offset_h), axis=-1)
    sample = np.broadcast_to(sample[:,:,np.newaxis], (B,N,4)) > 0.5
    
    anchor_offset = np.where(sample, offset, np.zeros_like(offset))
    anchor_mask = np.where(sample, np.ones_like(offset), np.zeros_like(offset))
    
    if flatten:
        anchor_offset = anchor_offset.reshape((B,-1))
        anchor_mask = anchor_mask.reshape((B,-1))
        
    return anchor_mask, anchor_offset


def get_label(anchor, gt_label_offset, cls_pred, match_threshould=0.5):

    gt = gt_label_offset[:,:,1:]
    gt_cls = gt_label_offset[:,:,0]
    cls_pred = cls_pred.asnumpy() 
    iou = calIOU(anchor, gt)
    mat = match(iou, threshould=match_threshould, share_max=False)
    samp = sample(mat, cls_pred, iou, ratio=3, min_sample=0, threshold=0.5, do=True)
    label_cls, label_mask = label_box_cls(mat, samp, gt_cls, ignore_label=-1)
    anchor_mask, anchor_offset = label_offset(anchor, gt, mat, samp)

    return label_cls, label_mask, anchor_offset, anchor_mask
