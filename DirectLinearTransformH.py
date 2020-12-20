import cv2
import numpy as np
import re
import math
from numpy.linalg import inv, norm


def __init__(self, orgs,corrs):
    self.orgs = orgs
    self.corrs = corrs

def average(org, corrs):
    averageOrg = np.zeros(2)
    averageCorr = np.zeros(2)
    for point in org:
        averageOrg[0] += point[0]
        averageOrg[1] += point[1]
    for point in corrs:
        averageCorr[0] += point[0]
        averageCorr[1] += point[1]
    averageOrg /= len(org)
    averageCorr /= len(corrs)

    return averageOrg, averageCorr


def scale(points, average):
    sumOfDist = 0.0
    for point in points:
        sumOfDist += norm(point - average)
    s = math.sqrt(2) * len(points) / (sumOfDist)
    return s


def matrixT(points, average):
    s = scale(points, average)
    tX = s * (-average[0])
    tY = s * (-average[1])
    T = np.float64([[s, 0, tX], [0, s, tY], [0, 0, 1]])
    return T


def normalize(points, T):
    normalizedPoints = np.zeros((len(points), 3))
    i = 0
    # ee = np.ones(len(points), order='C')
    # points=np.concatenate((points,ee))
    N=len(points)
    points=np.c_[points, np.ones(N)]
    # print(np.c_[points, np.ones(N)])
    for point in points:
        # print(T.shape)
        # print(point.shape)
        normalizedPoints[i] = np.transpose(T @ point)
        i += 1
    return normalizedPoints


def matrixA(normalizedOrg, normalizedCorr):
    A = np.zeros((len(normalizedOrg) * 2, 9))
    i = 0
    for index in range(0, len(A), 2):
        A[index] = np.float64([0, 0, 0, -normalizedOrg[i][0], -normalizedOrg[i][1], -1, normalizedCorr[i]
        [1] * normalizedOrg[i][0], normalizedCorr[i][1] * normalizedOrg[i][1], normalizedCorr[i][1]])
        A[index + 1] = np.float64([normalizedOrg[i][0], normalizedOrg[i][1], 1, 0, 0, 0, -normalizedCorr[i]
        [0] * normalizedOrg[i][0], -normalizedCorr[i][0] * normalizedOrg[i][1], -normalizedCorr[i][0]])
        i += 1
    return A


def computeH( src, dst):
    averageOrg, averageCorr = average(src, dst)
    orgT = matrixT(src, averageOrg)
    corrT = matrixT(dst, averageCorr)
    normalizedOrg = normalize(src, orgT)
    normalizedCorr = normalize(dst, corrT)
    A = matrixA(normalizedOrg, normalizedCorr)
    U, S, vT = cv2.SVDecomp(A)

    normalizedH = np.zeros((3, 3))
    i = 0
    for index in range(0, len(normalizedH)):
        normalizedH[index][0] = vT[-1][i]
        normalizedH[index][1] = vT[-1][i + 1]
        normalizedH[index][2] = vT[-1][i + 2]
        i += 3
    H = inv(corrT) @ normalizedH @ orgT
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

def main():
    flow_data = readFlow("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/flownet2-flow/000209.flo")

    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    print("u mean : " + str(np.mean(u)))
    print("v mean : " + str(np.mean(v)))
    print("u std : " + str(np.std(u)))
    print("v std : " + str(np.std(v)))
    print("u max : " + str(np.max(u)))
    print("u min : " + str(np.min(u)))
    print("v max : " + str(np.max(v)))
    print("v min : " + str(np.min(v)))

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
    H1 = computeH(p1, p2)
    np.set_printoptions(suppress=True)
    np.round_(H1, 4)
    print(H1)
    print(H1/H1[0,0])
    H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)
    print(H/H[0,0])
    return

if __name__ == '__main__':
    main()