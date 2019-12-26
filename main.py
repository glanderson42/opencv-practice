import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2


# globals
rightSlope = []
leftSlope = []
rightIntercept = []
leftIntercept = []


def getImages(path):
    image_files = os.listdir(path)

    image_list = []
    for i in range(0, len(image_files)):
        image_list.append(mpimg.imread(path + image_files[i]))

    return image_list


def displayImages(images, cmap = None):
    plt.figure(figsize=(40, 40))
    for i, image in enumerate(images):
        plt.subplot(3, 2, i + 1)
        plt.imshow(image, cmap)
        plt.autoscale(tight = True)

    plt.show()


def colorFilter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    low = np.array([0, 190, 0])
    up = np.array([255, 255, 255])
    
    yellower = np.array([10, 0, 90])
    yelupper = np.array([50, 255, 255])

    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, low, up)

    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked


def getRoi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])
    shape = np.array([
        [int(0), int(y)],
        [int(x), int(y)],
        [int(0.55 * x), int(0.6 * y)],
        [int(0.45 * x), int(0.6 * y)]
    ])

    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channels = image.shape[2]
        ignoreMaskColor = (255,) * channels
    else:
        ignoreMaskColor = 255

    cv2.fillPoly(mask, np.int32([shape]), ignoreMaskColor)
    maskedImage = cv2.bitwise_and(image, mask)
    
    return maskedImage


def getGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def getCanny(image):
    return cv2.Canny(getGrayscale(image), 50, 120)

def drawLines(image, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor = [0, 255, 0]
    leftColor = [255, 0, 0]
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1 - y2) / (x1 - x2)
            yintercept = y2 - (slope * x2)
            if slope > 0.3 and x1 > 500:
                rightSlope.append(slope)
                rightIntercept.append(yintercept)
            elif slope < -0.3 and x1 < 600:
                leftSlope.append(slope)
                leftIntercept.append(yintercept)
    
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    
    try:
        left_line_x1 = int((0.65 * image.shape[0] - leftavgIntercept) /
                leftavgSlope)

        left_line_x2 = int((image.shape[0] - leftavgIntercept) / leftavgSlope)

        right_line_x1 = int((0.65 * image.shape[0] - rightavgIntercept) / rightavgSlope)
        right_line_x2 = int((image.shape[0] - rightavgIntercept) / rightavgSlope)

        pts = np.array([
            [left_line_x1, int(0.65 * image.shape[0])],
            [left_line_x2, int(image.shape[0])],
            [right_line_x2, int(image.shape[0])],
            [right_line_x1, int(0.65 * image.shape[0])]
        ], np.int32)

        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(image,[pts],(0,0,255))
        cv2.line(image, (left_line_x1, int(0.65 * image.shape[0])),
                (left_line_x2, int(image.shape[0])), leftColor, 10)
        cv2.line(image, (right_line_x1, int(0.65 * image.shape[0])),
                (right_line_x2, int(image.shape[0])), rightColor, 10)
    except ValueError:
        pass


def getHoughtLines(image, rho, theta, threshold, minLineLen, maxLineGap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
            minLineLength=minLineLen, maxLineGap=maxLineGap)

    lineImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawLines(lineImage, lines)
    return lineImage


def linedetect(image):
    return getHoughtLines(image, 1, np.pi / 180, 10, 20, 100)


def weightedImage(image, initialImage, alfa=0.8, beta=1.0, gamma=0.0):
    return cv2.addWeighted(initialImage, alfa, image, beta, gamma)


def weightSum(inputSet):
    image = list(inputSet)
    return cv2.addWeighted(image[0], 1, image[1], 0.8, 0)


def processFrame(frame):
    interest = getRoi(frame)
    filteredFrame = colorFilter(interest)
    cannyImage = cv2.Canny(getGrayscale(filteredFrame), 50, 120)
    lines = getHoughtLines(cannyImage, 1, np.pi / 180, 10, 20, 5)
    weightedImg = cv2.addWeighted(lines, 1, frame, 0.8, 0)

    return weightedImg


if __name__ == '__main__':
    print(cv2.ocl.haveOpenCL())
    videoPath = sys.argv[1]
    video = cv2.VideoCapture(videoPath)
    orb = cv2.ORB_create()
    while(video.isOpened()):
        _, frame = video.read()

        if frame is None:
            break
            
        kps_frame = frame.copy()

        pts = cv2.goodFeaturesToTrack(np.mean(kps_frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]

        cv2.drawKeypoints(kps_frame, kps, kps_frame)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frame = cv2.putText(frame, str("fps: {}".format(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
        processedFrame = processFrame(frame)
        cv2.drawKeypoints(processedFrame, kps, processedFrame)
        
        # cv2.imshow('original', frame)
        cv2.imshow('processed', processedFrame)
        # cv2.imshow('feature points', kps_frame)
        
        key = cv2.waitKey(50)
        if key == 27:
            print('Pressed Esc')
            break

    video.release()
    cv2.destroyAllWindows()