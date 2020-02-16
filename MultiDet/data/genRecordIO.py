import xml.etree.ElementTree as ET
import cv2
import mxnet as mx
from tqdm import tqdm


obj_cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_path(voc_root):
    pathes = {}
    pathes['xml_root'] = voc_root + 'Annotations/'
    pathes['train_path'] = voc_root + 'ImageSets/Main/train.txt'
    pathes['val_path'] = voc_root + 'ImageSets/Main/val.txt'
    pathes['tv_path'] = voc_root + 'ImageSets/Main/trainval.txt'
    pathes['img_root'] = voc_root + 'JPEGImages/'
    return pathes


def get_cls(name):
    return obj_cls.index(name)


def get_clsname(cls):
    return obj_cls[cls]


def get_objs(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    size = root.find('size')
    w = size.find('width').text
    h = size.find('height').text

    return root.findall('object'), int(w), int(h)


def get_clspos(obj, w, h):
    name = obj.find('name')
    cls = get_cls(name.text)
    bndbox = obj.find('bndbox')

    xmin = int(bndbox.find('xmin').text) / w
    ymin = int(bndbox.find('ymin').text) / h
    xmax = int(bndbox.find('xmax').text) / w
    ymax = int(bndbox.find('ymax').text) / h
    assert xmin <= 1.0 and ymin <= 1.0 and xmax <= 1.0 and ymax <= 1.0, \
        'xmin %.2f, ymin %.2f, xmax %.2f, ymax %.2f' % (xmin, ymin, xmax, ymax)

    return [cls, xmin, ymin, xmax, ymax]


def get_label(xmlfile, pad=False, maxboxes=None, paddata=-1):
    objs, w, h = get_objs(xmlfile)

    result = []
    try:
        for obj in objs:
            result.extend(get_clspos(obj, w, h))
    except AssertionError:
        print(xmlfile)
        raise ValueError

    objs_num = len(objs)
    if pad:
        paddata = [paddata] * 5
        result.extend(paddata * (maxboxes - objs_num))

    result = [3, 5, objs_num] + result  

    return result  


def get_MaxNBox(file_path, xml_root):
    num = 0
    file = ''
    with open(file_path, 'r', encoding='utf8') as fp:
        for line in fp:
            xmlfile = xml_root + line.strip() + '.xml'
            objs, *_ = get_objs(xmlfile)
            if len(objs) > num:
                num = len(objs)
                file = xmlfile

    return num, file


def getXJpath(file, xroot, jroot):
    fp = open(file, 'r', encoding='utf8')
    for line in fp:
        xpath = xroot + line.strip() + '.xml'
        jpath = jroot + line.strip() + '.jpg'
        yield xpath, jpath
    fp.close()


def get_img(impath, resize=None):
    img = cv2.imread(impath) 
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def packLabelImg(label, img, quality=95):
    header = mx.recordio.IRHeader(0, label, id(label), 0)
    packio = mx.recordio.pack_img(header, img, quality=quality)
    return packio


def getRecordIO(filepath, maxbox, rec_name='voc', path='./',
                pad=True, resize=None, quality=95, voc_root='./'):
    rec_path = path + rec_name + '.rec'
    idx_path = path + rec_name + '.idx'

    recordio = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')
    getpath = getXJpath(filepath, xroot=get_path(voc_root)['xml_root'],
                        jroot=get_path(voc_root)['img_root'])

    for i, (xml, jpg) in tqdm(enumerate(getpath)):
        img = get_img(jpg, resize=resize)
        label = get_label(xml, pad=pad,
                          maxboxes=maxbox, paddata=-1)

        packio = packLabelImg(label, img, quality)
        recordio.write_idx(i, packio)

    recordio.close()


