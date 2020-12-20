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
    #print(p2)
    #print(len(p1))
    #H=getPerspectiveTransformMatrix(p1,p2)
    #H=cv2.getPerspectiveTransform(p1,p2)
    #H=H/H(0,0)
    H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)

    #print(dx)
    #print(sy)
    #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    #H = np.eye(3)
    # if H is None:
    #     H = np.eye(3)
    return H,p1,p2

def calculateHomography():
    H,p1,p2=homography("/home/nudlesoup/Research/flownet2-pytorch/homotest/homotrial.flo")

    np.set_printoptions(suppress=True)
    return H,p1,p2

def alignImages(im1, im2):
    # Convert images to grayscale
    # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    # orb = cv2.ORB_create(MAX_FEATURES)
    # keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)
    #
    # # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)
    #
    # # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    #
    # for i, match in enumerate(matches):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt
    #
    # # Find homography
    #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    h,p1,p2=calculateHomography()
    # imMatches = cv2.drawMatches(im1, p1, im2, p2, 1, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im2, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = "in01.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "in50.png"
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned8.png"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
