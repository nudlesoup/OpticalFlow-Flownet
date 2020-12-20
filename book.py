#!/usr/bin/env python

import cv2
import numpy as np
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

if __name__ == '__main__':
    # Read source image.
    # im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    # pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    pts_src=np.array([[328, 155], [307, 285], [285, 134], [325, 301]])
    # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    # pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    pts_dst = np.array([[324, 156], [303, 286], [279, 135], [319, 230]])
    # Calculate Homography
    #h, status = cv2.findHomography(pts_src, pts_dst)
    #h=cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    h=getPerspectiveTransformMatrix2(pts_src,pts_dst)
    np.set_printoptions(suppress=True)
    print(h)
    corners1=pts_src
    for i in range(len(corners1)):
        pt1 = np.array([corners1[i][0], corners1[i][1], 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(h, pt1)
        pt2 = pt2/pt2[2]
        print(pt2)
    # Warp source image to destination based on homography
    #im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    # Display images
    # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("Warped Source Image", im_out)
    #
    # cv2.waitKey(0)
