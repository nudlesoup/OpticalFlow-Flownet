import numpy as np
import sys,os
import torch
from torchvision import transforms
from PIL import Image
import cv2
TAG_CHAR = np.array([202021.25], np.float32)
def getPerspectiveTransformMatrix(p1, p2):
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
            # print(data.shape)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            x=np.resize(data, (int(h), int(w), 2))
            return x
def homography(flow_filename):
    flow_data = readFlow(flow_filename)
    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))
    print(u.shape)
    print(v.shape)
    print("v min : " + str(np.min(v)))

    offset = 5
    dx = u[::offset, ::offset]
    dy = v[::offset, ::offset]

    sy, sx = np.mgrid[:192:offset, :192:offset]
    # tx = sx + dx / offset
    # ty = sy + dy / offset
    tx= sx+dx
    ty = sy + dy
    aa = sx.flatten('F')
    bb = sy.flatten('F')
    cc = tx.flatten('F')
    dd = ty.flatten('F')
    # aa = aa.astype('float16')
    # bb = bb.astype('float16')
    # cc = cc.astype('float16')
    # dd = dd.astype('float16')
    p1=np.column_stack((aa, bb))
    p2=np.column_stack((cc, dd))
    # p1 = np.round_(p1, 4)
    # p2=np.round_(p2,4)
    # H = getPerspectiveTransformMatrix(p1,p2)
    H1,_ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    np.round_(H1, 4)
    print(H1/H1[0,0])
    np.set_printoptions(suppress=True)
    return H
if __name__ == '__main__':
    H= homography("/home/noodlesoup/Research/EgoPoseEstimation-IRI-DL/CustomDataset/ameya_basketball_lab_egoview/features/opticalflow/ameya_basketball_lab_egoview-0000.flo")
    print(H)