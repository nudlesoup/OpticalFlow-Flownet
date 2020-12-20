from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:17:29 2019
@author: BIEL
"""


import numpy as np
import random
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
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

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
    # H = np.round_(L, 4)
    print(H)
    # H=np.transpose(H)
    H=H/H[0,0]
    return H

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

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
    # u = cv2.normalize(flow_data[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    # v = cv2.normalize(flow_data[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    # u = u.astype('uint8')
    # v = v.astype('uint8')
    #print(u.shape)
    #print(v.shape)
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


    sx, sy = np.mgrid[10:500:25, 10:500:25]
    tx=sx+dx;
    ty=sy+dy;

    aa = sx.flatten('F')
    bb = sy.flatten('F')
    cc = tx.flatten('F')
    dd = ty.flatten('F')
    p1=np.column_stack((aa, bb))
    p2=np.column_stack((cc, dd))
    #print(sx)
    # print(np.max(tx))
    # print(np.max(sy))
    # print(np.max(ty))
    p1 = np.round_(p1, 4)
    p2=np.round_(p2,4)

    np.set_printoptions(suppress=True)
    #print(p2)
    print(len(p1))
    #H=getPerspectiveTransformMatrix2(p1,p2)
    #H=getPerspectiveTransformMatrix(p1,p2)
    srce_pts=np.reshape(p1,(len(p1),1,2))
    dest_pts=np.reshape(p2,(len(p2),1,2))
    H, mask = cv2.findHomography(srce_pts, dest_pts, cv2.RANSAC, 5.0)
    #H=H/H(0,0)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC,5.0)
    #H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    #print(dx)
    #print(sy)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    #H = np.eye(3)
    # if H is None:
    #     H = np.eye(3)
    return H,p1,p2,u,v

def calculateHomography():
    H,p1,p2,u,v=homography("/home/nudlesoup/Research/flownet2-pytorch/homotest/albert2-adjacent.flo")

    np.set_printoptions(suppress=True)
    return H,p1,p2,u,v

def line(img1,img2,corners1,H,u,v):

    img_draw_matches = cv2.hconcat([img1, img2])
    for i in range(len(corners1)):
        pt1 = np.array([corners1[i][0], corners1[i][1], 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(H, pt1)
        pt2 = pt2/pt2[2]
        # pt2=[u[corners1[i][0]-1,corners1[i][1]-1]+corners1[i][0],v[corners1[i][0]-1,corners1[i][1]-1]+corners1[i][1]]
        #print(pt2)
        end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
        cv2.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, random_color(), 2)

    out = "line-matches-cornersflow-adjacent2-newH.png"
    print("Saving aligned image : ", out)
    cv2.imwrite(out, img_draw_matches)


def alignImages(im1, im2):

    # # Find homography
    #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    h,p1,p2,u,v=calculateHomography()
    # imMatches = cv2.drawMatches(im1, p1, im2, p2, 1, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Use homography
    height, width, channels = im1.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    corners = [[328, 155], [307, 285], [285, 134], [325, 301], [362, 303]]
    corners2=[]
    line(im1, im2, corners, h,u,v)
    #line2(im1, im2, corners,corners2)
    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = "1.png"
    #print("Reading reference image : ", refFilename)
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "2.png"
    #print("Reading image to align : ", imFilename);
    im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    #print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im1, im2)

    # Write aligned image to disk.
    # outFilename = "line-aligned1m.png"
    # print("Saving aligned image : ", outFilename);
    # cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
