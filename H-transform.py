import cv2
import numpy as np
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

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
    # H=H/H[0,0]
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

# flow = readFlow("/home/nudlesoup/Research/flownet2-pytorch/rangetest/sports56/ex5/000209.flo")
im1 = np.asarray(cv2.imread("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/imxx376.png"))
im2 = np.asarray(cv2.imread("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/imxx377.png"))
# h = homography("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/flownet2-flow/000220.flo")
# h=np.array([[1 ,0.018678, -5.0169 ],[-0.044415 ,1.0023 ,10.818  ],[-0.0002988  , -0.00016687 , 1.0268]])
# h=np.array([[1,-0.003143,0.15981 ],[0.00047483,1.0002,0.26294],[1.643e-05,1.6544e-05,0.99734]])
# h=np.array([[1,-0.63423,41.207],[-0.45417,0.89077,41.915],[-0.0067873, -0.0086651,2.125]])
# h=np.array([[1, 0.00787 , -3.3288],[-0.033289 ,1.002  ,9.642],[-0.0002673 ,  -0.00014726 ,1.0265]])
# h=np.array([[1 ,-0.0062249 ,-2.753],[ 0.011399, 0.98135 , 2.5958],[ 0.00010459   ,-7.3017e-05  ,0.97954]])
# h=np.array([[1 ,-0.00072177 , 2.274],[ 0.0062482 ,1.0063 ,2.2497],[2.4326e-06 , -2.6015e-05  ,1.0081]])
# h = np.array([[1 ,-0.00033516 ,4.0573], [-0.0015442 ,1.0006 ,-1.9117], [3.292e-05  , -2.8085e-05  , 1.0016]])
# h = np.array([[1 , -0.001918 ,-1.7597], [-0.0010272  ,0.9996 ,   -0.20355], [2.9412e-06, 1.4714e-05 ,0.99901]])
# h = np.array([[1  ,-0.0014957, -1.2813], [0.00045335  ,1.0013  ,1.4906], [ -1.9892e-05 ,-2.0716e-05  ,1.0027]])
# h = np.array([[1 , 0.00091044 ,-0.26349], [-0.00092728 ,0.99806 ,-0.93464], [ 1.631e-05 , 6.5035e-09  , 0.99833]])
# h = np.array([[1  ,0.0030916 ,0.78766], [-0.00036827 , 1.0022 ,-0.20273], [ 8.9762e-06,  7.1003e-07 ,  1.0006]])
# h = np.array([[1 , 0.0049351 ,-2.6664], [0.0055095 ,0.99949  ,0.84363], [ 6.025e-06 , 5.5459e-05, 0.99352]])
h = np.array([[ 1. ,0.0018589  , 2.98964623],  [-0.02627923 , 1.02775027 , 1.40915763],  [-0.00018039,  0.00001928 , 1.03720317]])

np.set_printoptions(suppress=True)
np.round_(h, 4)
print(h)
warp = cv2.warpPerspective(im1, h, (192, 192))

# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex4/im1*255.jpg", im1*255)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/inference/imxx1-flow-warping.jpg", wrap1)
#cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/warping-fake.jpg", wrap1_fake)

# imageA=wrap1
# imageB=im2
imageA=im2
imageB=warp
# cv2.imwrite("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/offset-ransac-H-365.png", imageB)
# cv2.imwrite("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/groundtruth-H-436.png", imageB)
# cv2.imwrite("/home/noodlesoup/Research/flownet2-pytorch/rangetest/sports56/ex210/offset-you2meDLT-H-365.png", imageB)

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/original-frame2-0.jpg", imageA)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/modified-frame2-0.jpg", imageB)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/diff-0.jpg", diff)
# invert = cv2.bitwise_not(diff)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/diff-invert-0.jpg", invert)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/inference/unet.jpg", thresh)
