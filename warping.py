import cv2
import numpy as np
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

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
            x=x
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

flow = readFlow("/home/nudlesoup/Research/flownet2-pytorch/rangetest/flownet2-catch37/flow/000204.flo")
im1 = np.asarray(cv2.imread("/home/nudlesoup/Research/flownet2-pytorch/rangetest/inference/imxx205.png"))
im2 = np.asarray(cv2.imread("/home/nudlesoup/Research/flownet2-pytorch/rangetest/inference/imxx206.png"))

wrap1 = warp_flow(im1, flow)
wrap1_fake = warp_flow(im1, np.zeros_like(flow))

# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex4/im1*255.jpg", im1*255)
# cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/inference/imxx1-flow-warping.jpg", wrap1)
#cv2.imwrite("/home/nudlesoup/Research/flownet2-pytorch/rangetest/ex3/ssim/flownetSD400/warping-fake.jpg", wrap1_fake)

# imageA=wrap1
# imageB=im2
imageA=im1
imageB=im2
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
