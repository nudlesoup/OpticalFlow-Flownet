import numpy as np
import cv2
import random
import torch
from PIL import Image
from torchvision import transforms


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def opticalFlow(frame1, frame2, maxCorners, qualityLevel, show=False):
    # frames enter as numpy arrays
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=qualityLevel,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (maxCorners, 3))

    # Take first frame and find corners in it

    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1.copy())

    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    #    p0 = good_new.reshape(-1,1,2)

    if show:
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame2 = cv2.circle(frame2, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame2, mask)

        f = 1
        cv2.imshow('test', cv2.resize(img, (0, 0), fx=f, fy=f))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imwrite('soft-test-2.jpg', img)

    return good_old, good_new

def line(img1,img2,corners1,h):
    img_draw_matches = cv2.hconcat([img1, img2])
    for i in range(len(corners1)):
        pt1 = np.array([corners1[i][0], corners1[i][1], 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(h, pt1)
        pt2 = pt2/pt2[2]
        end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
        cv2.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, random_color(), 2)

    out = "softopotical-matches-mycorrs.png"
    print("Saving aligned image : ", out)
    cv2.imwrite(out, img_draw_matches)

def homography(frame1, frame2, maxCorners=100, qualityLevel=0.7, show=False):
    p0, p1 = opticalFlow(frame1, frame2, maxCorners=maxCorners, qualityLevel=qualityLevel, show=True)
    H, _ = cv2.findHomography(p0, p1, method=cv2.RANSAC)
    if H is None:
        H = np.eye(3)
    return H, p0, p1

if __name__ == '__main__':
    # Read reference image
    refFilename = "test1.png"
    print("Reading reference image : ", refFilename)
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "test2.png"
    print("Reading image to align : ", imFilename);
    im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    H,p0,p1 = homography(im1, im2)
    #print(p0)
    #print(p1)
    corners = [[318, 156], [271, 136], [298, 286], [316, 302], [352, 302]]
    line(im1,im2,corners,H)
    #print(H)
    # # Write aligned image to disk.
    # outFilename = "line-aligned1m.png"
    # print("Saving aligned image : ", outFilename);
    # cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", H)