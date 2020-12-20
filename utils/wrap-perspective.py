# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:17:29 2019
@author: BIEL
"""


import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

TAG_CHAR = np.array([202021.25], np.float32)

def getPerspectiveTransformMatrix(p1, p2):
    matrixIndex = 0
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        # A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        # A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        #A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    #print(A)
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    L=Vh[8,:].reshape(3, 3)
    H=L/L[0,0]

    #      # A[matrixIndex] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
    #     # A[matrixIndex + 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    #     A[matrixIndex] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    #     A[matrixIndex + 1] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
    # U, s, V = np.linalg.svd(A, full_matrices=True)
    # matrix = V[:,9].reshape(3, 3).transpose()
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
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            x=np.resize(data, (int(h), int(w), 2))
            return x

def homography(flow_filename):
    flow_data = readFlow(flow_filename)
    #print(flow_data.shape)
    u = flow_data[:, :,0]
    v = flow_data[:, :,1]
    #print(u.shape)
    #print(v.shape)
    dx=np.zeros((17,17))
    dy = np.zeros((17, 17))
    a=0
    for i in range(9,90,5):
        b=0
        for j in range(9,90,5):
            dx[a,b]=u[i,j]
            dy[a,b]=v[i,j]
            b=b+1
        a=a+1


    sy, sx = np.mgrid[10:95:5, 10:95:5]
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
    #print(p2)
    #print(len(p1))
    H=getPerspectiveTransformMatrix(p1,p2)
    #H=cv2.getPerspectiveTransform(p1,p2)
    #H=H/H(0,0)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)

    #print(dx)
    #print(sy)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    #H = np.eye(3)
    # if H is None:
    #     H = np.eye(3)
    return H

def calculateHomography():
    H=homography("/home/nudlesoup/Research/flownet2-pytorch/homotest/000000.flo")

    np.set_printoptions(suppress=True)
    return H