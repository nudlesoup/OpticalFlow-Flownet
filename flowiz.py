# Converts Flow .flo files to Images

# Author : George Gach (@georgegach)
# Date   : July 2019

# Adapted from the Middlebury Vision project's Flow-Code
# URL    : http://vision.middlebury.edu/flow/

import numpy as np
import os
import errno
from tqdm import tqdm
from PIL import Image
import io

TAG_FLOAT = 202021.25
flags = {
    'debug': False
}

def _normalize_flow(flow):
    UNKNOWN_FLOW_THRESH = 1e9
    # UNKNOWN_FLOW = 1e10

    height, width, nBands = flow.shape
    if not nBands == 2:
        raise AssertionError("Image must have two bands. [{h},{w},{nb}] shape given instead".format(
            h=height, w=width, nb=nBands))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Fix unknown flow
    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([-1, np.max(rad)])

    if flags['debug']:
        print("Max Flow : {maxrad:.4f}. Flow Range [u, v] -> [{minu:.3f}:{maxu:.3f}, {minv:.3f}:{maxv:.3f}] ".format(
            minu=minu, minv=minv, maxu=maxu, maxv=maxv, maxrad=maxrad
        ))

    eps = np.finfo(np.float32).eps
    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    return u, v


def _flow2color(flow):

    u, v = _normalize_flow(flow)
    img = _compute_color(u, v)

    # TO-DO
    # Indicate unknown flows on the image
    # Originally done as
    #
    # IDX = repmat(idxUnknown, [1 1 3]);
    # img(IDX) = 0;

    return img


def _flow2uv(flow):
    u, v = _normalize_flow(flow)
    uv = (np.dstack([u, v])*127.999+128).astype('uint8')
    return uv


def _save_png(arr, path):
    # TO-DO: No dependency
    Image.fromarray(arr).save(path)


def convert_from_file(path, mode='RGB'):
    return convert_from_flow(read_flow(path), mode)


def convert_from_flow(flow, mode='RGB'):
    if mode == 'RGB':
        return _flow2color(flow)
    if mode == 'UV':
        return _flow2uv(flow)

    return _flow2color(flow)


def convert_files(files, outdir=None):
    if outdir != None and not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
            print("> Created directory: " + outdir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    t = tqdm(files)
    for f in t:
        image = convert_from_file(f)

        if outdir == None:
            path = f + '.png'
            t.set_description(path)
            _save_png(image, path)
        else:
            path = os.path.join(outdir, os.path.basename(f) + '.png')
            t.set_description(path)
            _save_png(image, path)