
import numpy as np
import sys,os

import torch
from torchvision import transforms
from PIL import Image
import cv2
import random


# TAG_CHAR = np.array([202021.25], np.float32)
# def getPerspectiveTransformMatrix2(p1, p2):
#     matrixIndex = 0
#     A=[]
#     for i in range(0, len(p1)):
#         x, y = p1[i][0], p1[i][1]
#         u, v = p2[i][0], p2[i][1]
#         A.append( [-x, -y, -1, 0, 0, 0, u * x, u * y, u])
#
#     for i in range(0, len(p1)):
#         x, y = p1[i][0], p1[i][1]
#         u, v = p2[i][0], p2[i][1]
#         A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
#
#     A = np.asarray(A)
#
#     U, S, Vh = np.linalg.svd(A)
#     np.set_printoptions(suppress=True)
#     #print(Vh)
#     L = Vh[-1,:]
#
#     H = np.reshape(L,(3, 3))
#     H=H/H[0,0]
#     return H
#
# def readFlow(fn):
#     """ Read .flo file in Middlebury format"""
#     with open(fn, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         if 202021.25 != magic:
#             print('Magic number incorrect. Invalid .flo file')
#             return None
#         else:
#             w = np.fromfile(f, np.int32, count=1)
#             h = np.fromfile(f, np.int32, count=1)
#             #print('Reading %d x %d flo file\n' % (w, h))
#             data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
#             # Reshape data into 3D array (columns, rows, bands)
#             # The reshape here is for visualization, the original code is (w,h,2)
#             x=np.resize(data, (int(h), int(w), 2))
#             return x
# def homography2(flow_filename):
#     flow_data = readFlow(flow_filename)
#     #print(flow_data.shape)
#     u = flow_data[:, :,0]
#     v = flow_data[:, :,1]
#     # UNKNOW_FLOW_THRESHOLD = 1e7
#     # pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
#     # pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
#     # idx_unknown = (pr1 | pr2)
#     # u[idx_unknown] = v[idx_unknown] = 0
#     #
#     # # get max value in each direction
#     # maxu = -999.
#     # maxv = -999.
#     # minu = 999.
#     # minv = 999.
#     # maxu = max(maxu, np.max(u))
#     # maxv = max(maxv, np.max(v))
#     # minu = min(minu, np.min(u))
#     # minv = min(minv, np.min(v))
#     #
#     # rad = np.sqrt(u ** 2 + v ** 2)
#     # maxrad = max(-1, np.max(rad))
#     # u = u / maxrad + np.finfo(float).eps
#     # v = v / maxrad + np.finfo(float).eps
#     # u = cv2.normalize(flow_data[..., 0], None, -10, 10, cv2.NORM_MINMAX)
#     # v = cv2.normalize(flow_data[..., 1], None, -10, 10, cv2.NORM_MINMAX)
#     print(u.shape)
#     print(v.shape)
#     dx=np.zeros((37,37))
#     dy = np.zeros((37, 37))
#     a=0
#     for i in range(9,190,5):
#         b=0
#         for j in range(9,190,5):
#             dx[a,b]=u[i,j]
#             dy[a,b]=v[i,j]
#             b=b+1
#         a=a+1
#
#     #print(dx)
#     sy, sx = np.mgrid[10:191:5, 10:191:5]
#
#     tx=sx+dx;
#     ty=sy+dy;
#
#     aa = sx.flatten('F')
#     bb = sy.flatten('F')
#     cc = tx.flatten('F')
#     dd = ty.flatten('F')
#     p1=np.column_stack((aa, bb))
#     p2=np.column_stack((cc, dd))
#     p1 = np.round_(p1, 4)
#     p2=np.round_(p2,4)
#     np.set_printoptions(suppress=True)
#     np.set_printoptions(suppress=True)
#     #print(sx)
#     # print(dd)
#     # print(np.max(cc))
#     # print(np.max(dd))
#     H=getPerspectiveTransformMatrix2(p1,p2)
#     #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
#     return H
#
# def homography(flow_filename):
#     flow_data = readFlow(flow_filename)
#     #print(flow_data.shape)
#     u = flow_data[:, :,0]
#     v = flow_data[:, :,1]
#
#     dx=np.zeros((20,20))
#     dy = np.zeros((20, 20))
#     a=0
#     for i in range(9,500,25):
#         b=0
#         for j in range(9,500,25):
#             dx[a,b]=u[i,j]
#             dy[a,b]=v[i,j]
#             b=b+1
#         a=a+1
#
#
#     sy, sx = np.mgrid[10:500:25, 10:500:25]
#     tx=sx+dx;
#     ty=sy+dy;
#
#     aa = sx.flatten('F')
#     bb = sy.flatten('F')
#     cc = tx.flatten('F')
#     dd = ty.flatten('F')
#     p1=np.column_stack((aa, bb))
#     p2=np.column_stack((cc, dd))
#     p1 = np.round_(p1, 4)
#     p2=np.round_(p2,4)
#     np.set_printoptions(suppress=True)
#     np.set_printoptions(suppress=True)
#     #print(sx)
#     # print(dd)
#     # print(np.max(cc))
#     # print(np.max(dd))
#     H=getPerspectiveTransformMatrix2(p1,p2)
#     #H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
#     return H

TAG_CHAR = np.array([202021.25], np.float32)


def findFeatures(img):
    print("Finding Features...")
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints, outImage = None, color=(255,0,0))
    # cv2.imwrite('sift_keypoints.png', img)

    return keypoints, descriptors
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    # matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    # cv2.imwrite('Matches.png', matchImg)
    return matches

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
    offset = 5
    dx = u[::offset, ::offset]
    dy = v[::offset, ::offset]

    sy, sx = np.mgrid[:192:offset, :192:offset]
    tx = sx + dx / offset
    ty = sy + dy / offset
    # tx= sx+dx
    # ty = sy + dy
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
    H,_ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    H=H/H[0,0]
    # np.round_(H1, 4)
    # print(H1/H1[0,0])
    # np.set_printoptions(suppress=True)
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

            u = x[:, :, 0]
            v = x[:, :, 1]
            print("u mean : " + str(np.mean(u)))
            print("v mean : " + str(np.mean(v)))
            print("u std : " + str(np.std(u)))
            print("v std : " + str(np.std(v)))
            print("u max : " + str(np.max(u)))
            print("u min : " + str(np.min(u)))
            print("v max : " + str(np.max(v)))
            print("v min : " + str(np.min(v)))
            return x

if __name__ == '__main__':
    refFilename = "/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/imxx376.png"
    # print("Reading reference image : ", refFilename)
    img1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/imxx377.png"
    # print("Reading image to align : ", imFilename);
    img2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    # print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.

    H = homography("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/flownet2-flow/000375.flo")
    #H=H=np.array([[1 ,21.389, -1971.4],[22.821, 1.5749 , -2115.9 ],[0.36405 ,  0.35232 , -56.742 ]])
    #H=np.array([[1 ,-0.0024963, 0.0075818],[0.00060423 ,0.99798 ,-0.012562 ],[2.2078e-05  , -1.6688e-05 , 0.99796 ]])
    #H=np.array([[1, -2.8116, 262.65],[-2.8555 ,0.94704 ,267.92],[-0.04713  , -0.044603 ,8.4566]])
    # H = np.array([[1, 0.00787, -3.3288], [-0.033289, 1.002, 9.642], [-0.0002673, -0.00014726, 1.0265]])
    # H = np.array([[1, 0.00091044, -0.26349], [-0.00092728, 0.99806, -0.93464], [1.631e-05, 6.5035e-09, 0.99833]])
    # H = np.array([[1, 0.0030916, 0.78766], [-0.00036827, 1.0022, -0.20273], [8.9762e-06, 7.1003e-07, 1.0006]])
    H = np.array([[1, 0.0030916, 0.78766], [-0.00036827, 1.0022, -0.20273], [8.9762e-06, 7.1003e-07, 1.0006]])

    H=H.T
    print(H)
    img_draw_matches = cv2.hconcat([img1, img2])
    # pts_src = np.array([[30, 30], [60, 30], [90, 30], [120, 30],[150, 30],[180, 30],[30, 90],[60, 90],[90, 90],[120, 90],[150, 90],[180, 90],[30, 120],[60, 120],[90, 120],[120, 120],[150, 120],[180, 120],[30, 150],[60, 150],[90, 150],[120, 150],[150, 150],[180, 150],[30, 180],[60, 180],[90, 180],[120, 180],[150, 180],[180, 180]])
    # pts_src1 = np.array([[183, 334], [216, 328], [351, 420], [378, 413], [196, 463]])
    # pts_dest = np.array([[170, 277], [207, 271], [343, 411], [368, 402], [188, 457]])
    #H = getPerspectiveTransformMatrix2(pts_src1, pts_dest)
    #pts_src = np.array([[183, 334], [216, 328], [351, 420], [378, 413], [196, 463],[10, 10],[70, 35],[30, 60],[200, 85],[50, 110],[180, 150],[100, 190],[300, 230],[150, 270],[350, 310],[170, 350],[390, 390],[410, 410],[430, 430],[470, 470] ])
    #pts_src=np.zeros(20)
    # pts_src = np.array([[30,30],[45,45],[60,60],[90,90],[120,120],[150,150],[180,180]])
    kp1, desc1 = findFeatures(img1)

    kp2, desc2 = findFeatures(img2)
    correspondenceList = []
    keypoints = [kp1, kp2]
    matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt

        correspondenceList.append([x1, y1,1])

    corrs = np.matrix(correspondenceList)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    # corners1 = pts_src
    for i in range(len(correspondenceList)):
        # pt1 = np.array([corners1[i][0], corners1[i][1], 1])
        pt1 = correspondenceList[i]
        # pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(H, pt1)
        pt2 = pt2 / pt2[2]
        print(pt2)
        end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
        # cv2.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, random_color(), 2)
        x1= correspondenceList[i][0]
        y1= correspondenceList[i][1]
        x2= pt2[0]
        y2= pt2[1]
        cv2.circle(img_draw_matches, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(img_draw_matches, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
        cv2.line(img_draw_matches, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)
    file = "/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/line-sift-gtTranspose-375.png"
    #print("Saving aligned image : ", out)
    # cv2.imwrite(out, img_draw_matches)
    cv2.imwrite(file, img_draw_matches)
    #H = homography("/home/nudlesoup/Research/Ameya-You2me/flow/albert_basketball_indoor2_egoview/albert_basketball_indoor2_egoview-000000.flo")
    np.set_printoptions(suppress=True)
    np.round_(H, 4)
    print(H)
