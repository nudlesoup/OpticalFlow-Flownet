
import numpy as np
import sys,os

import torch
from torchvision import transforms
from PIL import Image
import cv2


TAG_CHAR = np.array([202021.25], np.float32)
def getPerspectiveTransformMatrix2(p1, p2):
    matrixIndex = 0
    A=[]
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append( [-x, -y, -1, 0, 0, 0, u * x, u * y, u])

    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A)
    np.set_printoptions(suppress=True)
    #print(Vh)
    L = Vh[-1,:]

    H = np.reshape(L,(3, 3))
    H=H/H[0,0]
    return H
def get_flow():
    frame1 = cv2.imread('kinect-image/imxx351.png')
    frame2 = cv2.imread('kinect-image/imxx352.png')
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    u = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    print(u.shape)
    dx = np.zeros((37, 37))
    dy = np.zeros((37, 37))
    a = 0
    for i in range(9, 190, 5):
        b = 0
        for j in range(9, 190, 5):
            dx[a, b] = u[i, j]
            dy[a, b] = v[i, j]
            b = b + 1
        a = a + 1

    # print(dx)
    sy, sx = np.mgrid[10:191:5, 10:191:5]

    tx = sx + dx;
    ty = sy + dy;

    aa = sx.flatten('F')
    bb = sy.flatten('F')
    cc = tx.flatten('F')
    dd = ty.flatten('F')
    p1 = np.column_stack((aa, bb))
    p2 = np.column_stack((cc, dd))
    p1 = np.round_(p1, 4)
    p2 = np.round_(p2, 4)
    np.set_printoptions(suppress=True)
    np.set_printoptions(suppress=True)
    # print(sx)
    # print(dd)
    # print(np.max(cc))
    # print(np.max(dd))
    H = getPerspectiveTransformMatrix2(p1, p2)
    return H

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)

            print('Reading %d x %d flo file\n' % (w, h))

            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            x=np.resize(data, (int(h), int(w), 2))
            # TRIAl
            # np_flow = np.fromfile(f, np.float32)
            # print("xx : " + str(np_flow.shape))
            # #tag = np.fromfile(f, np.float32, count=1)
            # width = np.fromfile(f, np.int32, count=1)
            # height = np.fromfile(f, np.int32, count=1)
            #
            # print('tag', 'width', width, 'height', height)
            #
            # nbands = 2
            # tmp = np.fromfile(f, np.float32, count=nbands * width * height)
            # flow = np.resize(tmp, (int(height), int(width), int(nbands)))
            print(x.shape)
            print(x[0][0])
            return x

def homography2(flow_filename):
    flow_data = readFlow(flow_filename)
    #print(flow_data.shape)
    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    print(str(np.mean(u)) + " " + str(np.std(u)))
    print(str(np.mean(v)) + " " + str(np.std(v)))
    u = cv2.normalize(flow_data[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(flow_data[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    UNKNOW_FLOW_THRESHOLD = 250
    pr1 = abs(u) < UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    # idx_unknown = (pr1 | pr2)
    # u[idx_unknown] = v[idx_unknown] = 0
    #
    # # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))
    #print(np.transpose(np.nonzero(pr1>0)))
    print("maxu "+ str(maxu) + " maxv " + str(maxv) + " minu " + str(minu) + " minv " + str(minv))

    # # rad = np.sqrt(u ** 2 + v ** 2)
    # # maxrad = max(-1, np.max(rad))
    # # u = u / maxrad + np.finfo(float).eps
    # # v = v / maxrad + np.finfo(float).eps
    # print(u.shape)
    # print(v.shape)
    dx=np.zeros((37,37))
    dy = np.zeros((37, 37))
    a=0
    for i in range(9,190,5):
        b=0
        for j in range(9,190,5):
            dx[a,b]=u[i,j]
            dy[a,b]=v[i,j]
            b=b+1
        a=a+1

    #print(dx)
    sy, sx = np.mgrid[10:191:5, 10:191:5]

    tx=sx+dx;
    ty=sy+dy;

    aa = sx.flatten('F')
    bb = sy.flatten('F')
    cc = tx.flatten('F')
    dd = ty.flatten('F')
    p1=np.column_stack((aa, bb))
    p2=np.column_stack((cc, dd))
    p1 = np.round_(p1, 4)
    p2=np.round_(p2,4)
    np.set_printoptions(suppress=True)
    np.set_printoptions(suppress=True)
    H=getPerspectiveTransformMatrix2(p1,p2)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    return H



H=get_flow()
# H = homography2("/home/nudlesoup/Research/flownet2-pytorch/kinect-image/kinect-0.flo")

# np.set_printoptions(suppress=True)
# np.round_(H, 4)
print(H)