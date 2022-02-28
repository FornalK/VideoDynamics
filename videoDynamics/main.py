import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def extractImages(videoPath, numOfFrames):
    extractedFrames = []
    vidcap = cv2.VideoCapture(videoPath)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    scope = round(duration)

    success, frame = vidcap.read()
    success = True
    if not numOfFrames:
        count = 1
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count * 1000 - 500))
            success, frame = vidcap.read()
            if success:
                extractedFrames.append(frame[...,::-1])
            count = count + 1
    else:
        pass
    return extractedFrames

"""
def toGrayscale(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for i in range(numOfFrames):
        frameInGrayscale = color.rgb2gray(frames[i])
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale
"""

def toGrayscale(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for i in range(numOfFrames):
        frameInGrayscale = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale

def arrayToVector(frames):
    numOfFrames = len(frames)
    vectors = []
    for i in range(numOfFrames):
        vectors.append(frames[i].flatten())
    return vectors

def normalize(frames):
    numOfFrames = len(frames)
    normalizedFrames = []
    I = frames[0].shape[0]
    for k in range(numOfFrames):
        Xmax = np.max(frames[k])
        Xmin = np.min(frames[k])
        normalizedFrame = np.empty((I))
        for i in range(I):
            normalizedFrame[i] = (frames[k][i] - Xmin) / (Xmax - Xmin)
        normalizedFrames.append(normalizedFrame)
    return normalizedFrames

def measureDistance(frames):
    numOfFrames = len(frames)
    partial_d = []
    I = frames[0].shape[0]
    for k in range(numOfFrames - 1):
        sum = 0
        for i in range(I):
            temp = abs(frames[k][i] - frames[k + 1][i])
            sum += temp
        partial_d.append(sum)
    return np.mean(partial_d)

videoPath = 'film1_1080p.mp4'
numOfFrames = False # jeśli false to domyślnie wyciągana jest z filmu jedna klatka na sekundę

frames1 = extractImages(videoPath, numOfFrames)
im1 = cv2.imread("1.png")[...,::-1]
im2 = cv2.imread("2.png")[...,::-1]
im3 = cv2.imread("3.png")[...,::-1]
im4 = cv2.imread("4.png")[...,::-1]

plt.imshow(frames1[0])
plt.show()

frames2 = [im1, im2]
frames3 = [im3, im4]
array = [frames1, frames2, frames3]

"""
for arr in array:
    frames = toGrayscale(arr)
    frames = normalize(frames)
    d = measureDistance(frames)
    print(d)
"""

prz1 = toGrayscale(frames2)
prz2 = arrayToVector(prz1)
prz3 = normalize(prz2)
prz4 = measureDistance(prz3)
print(prz4)








