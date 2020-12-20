
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
            #print('Reading %d x %d flo file\n' % (w, h))
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # print(data.shape)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            x=np.resize(data, (int(h), int(w), 2))
            return x
def readFlow2(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print('Reading %d x %d flo file' % (w, h))
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
    return data2D

def homography2(flow_filename):
    flow_data = readFlow(flow_filename)
    #print(flow_data.shape)
    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    print(u.shape)
    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))
    dx=np.zeros((20,20))
    dy = np.zeros((20, 20))
    a=0
    for i in range(9,500,25):
        b=0
        for j in range(9,500,25):
            dx[a,b]=u[i,j]
            dy[a,b]=v[i,j]
            b=b+1
        a=a+1


    sy, sx = np.mgrid[10:500:25, 10:500:25]
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

    H=getPerspectiveTransformMatrix(p1,p2)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    return H

def homography3(flow_filename):
    flow_data = readFlow(flow_filename)
    #print(flow_data.shape)
    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    print(u.shape)
    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))
    dx=np.zeros((50,50))
    dy = np.zeros((50, 50))
    a=0
    for i in range(9,501,10):
        b=0
        for j in range(9,501,10):
            dx[a,b]=u[i,j]
            dy[a,b]=v[i,j]
            b=b+1
        a=a+1


    sy, sx = np.mgrid[10:501:10, 10:501:10]
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

    #H=getPerspectiveTransformMatrix(p1,p2)
    H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    return H

def homography(flow_filename):
    flow_data = readFlow(flow_filename)

    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    # u = cv2.normalize(flow_data[..., 0], None, -10, 10, cv2.NORM_MINMAX)
    # v = cv2.normalize(flow_data[..., 1], None, -10, 10, cv2.NORM_MINMAX)
    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))

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
    H1=getPerspectiveTransformMatrix(p1,p2)
    np.set_printoptions(suppress=True)
    np.round_(H1, 4)
    print(H1)
    H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    return H
def checkflowvalues(flow_filename):
    flow_data = readFlow2(flow_filename)

    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    # print(u.shape)
    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))



if __name__ == '__main__':
    #H = homography2("/home/nudlesoup/Research/flownet2-pytorch/rangetest/kdata/K.flo")
    # H = homography("/home/nudlesoup/Research/flownet2-pytorch/rangetest/flownet2-cat3ch37.flo")
    # H = homography("/home/nudlesoup/Research/flownet2-pytorch/rangetest/sports56/flownet2-flow/000000.flo")
    # H = homography("/home/nudlesoup/Research/flownet2-pytorch/rangetest/sports56/Unet/normal/000.flo")
    checkflowvalues("/home/nudlesoup/Research/flownet2-pytorch/rangetest/flying chair/0/0000001-gt.flo")
    checkflowvalues("/home/nudlesoup/Research/flownet2-pytorch/rangetest/flying chair/0/flownet2-1.flo")
    # np.set_printoptions(suppress=True)
    # np.round_(H, 4)
    # print(H)
    # M=H/H[0,0]
    # print(M)
