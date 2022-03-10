import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import random
import pandas as pd
import xlwt
from tempfile import TemporaryFile

"""
def toGrayscaleAndNormalize(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for i in range(numOfFrames):
        frameInGrayscale = color.rgb2gray(frames[i])
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale
"""

"""
def dividingIntoScenes(videoPath, sampling, tabOfScenesTimestamps):
    scenes = []
    vidcap = cv2.VideoCapture(videoPath)

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    success, frame = vidcap.read()
    success = True

    shift = 0.25
    for k in range(len(tabOfScenesTimestamps)):
        scene = []
        i = tabOfScenesTimestamps[k] + shift
        if (k == len(tabOfScenesTimestamps) - 1):
            endTimestamp = duration
        else:
            endTimestamp = tabOfScenesTimestamps[k + 1]
        while i < endTimestamp:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
            success, frame = vidcap.read()
            if success:
                scene.append(frame[..., ::-1])
                #plt.imshow(frame[..., ::-1])
                #plt.show()
            else:
                raise Exception
            print("scene num:", k + 1, "\tframe timestamp:", i)
            i += sampling
        scenes.append(scene)

    return (scenes, duration)
"""

def toGrayscaleAndNormalize(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for k in range(numOfFrames):
        R = frames[k][:,:,0].astype(int)
        G = frames[k][:,:,1].astype(int)
        B = frames[k][:,:,2].astype(int)
        frameInGrayscale = np.round_((R + G + B) / 3).astype(int) / 255
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale

def measureDistance(frames):
    numOfFrames = len(frames)
    partial_d = []
    for i in range(numOfFrames - 1):
        partial_d.append(np.mean(np.abs(frames[i] - frames[i + 1])))
    #print(partial_d)
    return np.mean(partial_d)

def dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes):
    vidcap = cv2.VideoCapture(videoPath)
    scenes = []
    frames = []
    dataPackage = {}

    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        frame_ready, frame = vidcap.read()
        if frame_ready:
            frames.append(frame[..., ::-1])
            pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    dataPackage['number of frames'] = frame_count
    dataPackage['duration'] = duration
    dataPackage['fps'] = fps
    dataPackage['number of scenes'] = len(tabOfScenesFirstFramesIndexes)
    dataPackage['scenes per second'] = len(tabOfScenesFirstFramesIndexes) / duration
    dataPackage['sample size (%)'] = sampling

    for k in range(len(tabOfScenesFirstFramesIndexes)):
        if k == len(tabOfScenesFirstFramesIndexes) - 1:
            scenes.append(frames[tabOfScenesFirstFramesIndexes[k]:])
        else:
            scenes.append(frames[tabOfScenesFirstFramesIndexes[k]:tabOfScenesFirstFramesIndexes[k+1]])

    sampledScenes = []
    numOfFramesInEachScene = []
    numOfFramesInSample = []
    numberOfFramesTaken = []
    for scene in scenes:
        if (sampling > 100 or sampling < 0):
            raise
        numOfFramesInEachScene.append(len(scene))
        amount = round((sampling / 100) * len(scene))
        if amount == 0:
            amount = 1
        if amount == 1:
            middleFrameIndex = len(scene) // 2
            sampledScene = scene[middleFrameIndex]
            sampledScenes.append([sampledScene])
            numOfFramesInSample.append(1)
            numberOfFramesTaken.append(1)
        else:
            """
            step = round(len(scene) / amount)
            sampledScene = []
            for x in range(0, len(scene), step):
                sampledScene.append(scene[x])

            sampledScenes.append(sampledScene)
            numOfFramesInSample.append(amount)
            numberOfFramesTaken.append(len(sampledScene))
            """

            step = round(len(scene) / amount)
            sampledScene = []
            for x in range(0, len(scene), step):
                index = x+step // 2
                if index < len(scene):
                    sampledScene.append(scene[index])

            sampledScenes.append(sampledScene)

            numOfFramesInSample.append(amount)
            numberOfFramesTaken.append(len(sampledScene))

        dataPackage['number of frames in each scene'] = numOfFramesInEachScene
        dataPackage['number of frames in sample'] = numOfFramesInSample
        dataPackage['number of frames taken'] = numberOfFramesTaken

    return (sampledScenes, duration, dataPackage)

def main(videoPath, sampling, tabOfScenesFirstFramesIndexes=[]):
    if tabOfScenesFirstFramesIndexes != []:
        scenes, duration, dataPackage = dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes)
        internalScenes_d = []
        for scene in scenes:
            if len(scene) == 1:
                internalScenes_d.append(None)
            else:
                frames = toGrayscaleAndNormalize(scene)
                internalScenes_d.append(measureDistance(frames))
        print("mean internal scenes distance: ", internalScenes_d)

        averagedFrames = []
        for k, scene in enumerate(scenes):
            I, J, K = scene[0].shape
            averagedFrame = np.zeros((I, J, K), dtype=int)
            for frame in scene:
                averagedFrame += frame
            averagedFrame = np.round_(averagedFrame / len(scene)).astype(int)

            averagedFrames.append(averagedFrame)
            #plt.imshow(averagedFrame)
            #plt.show()

        externalScenes_d = []
        for k in range(len(averagedFrames) - 1):
            frames = toGrayscaleAndNormalize([averagedFrames[k], averagedFrames[k+1]])
            externalScenes_d.append(measureDistance(frames))

        print("distance between individual scenes", externalScenes_d)

        transitions = []
        for k, scene in enumerate(scenes):
            if (k == 0):
                transitions.append(scene[-1])
            elif (k == len(scenes) - 1):
                transitions.append(scene[0])
            else:
                transitions.append(scene[-1])
                transitions.append(scene[0])

        """
        transitions_d = []
        for k in range(0, len(transitions), 2):
            frames = toGrayscaleAndNormalize([transitions[k], transitions[k + 1]])
            transitions_d.append(measureDistance(frames))
        print("transitions", transitions_d)
        """

        sps = len(tabOfScenesFirstFramesIndexes) / duration  # scenes per second
        print("scenes per second:", sps)

        dataPackage['mean internal scenes distance'] = internalScenes_d
        dataPackage['distance between individual scenes'] = externalScenes_d

        return scenes, dataPackage
    else:
        pass #to-do

videoPath = 'videos/pizza.mp4'
test1 = [0, 76, 193, 280, 328]
test2 = [0, 29, 43, 72, 87, 120, 151, 173, 237]
tymbark = [0]
pizza = [0, 22, 49, 83, 113, 144, 165, 173, 181, 195, 206, 219, 232, 242, 254, 298 ]

"""
im1 = cv2.imread("1.png")[...,::-1]
im2 = cv2.imread("2.png")[...,::-1]
im3 = cv2.imread("3.png")[...,::-1]
im4 = cv2.imread("4.png")[...,::-1]
white = cv2.imread("white.png")[...,::-1]
black = cv2.imread("black.png")[...,::-1]
gray = cv2.imread("gray.png")[...,::-1]
frames2 = [im1, im2]
frames3 = [im3, im4]
frames4 = [black, white]
array = [frames2, frames3, frames4]

for arr in array:
    frames = toGrayscaleAndNormalize(arr)
    d = measureDistance(frames)
    print(d)

"""

"""
scenes, dataPackage = main(videoPath, sampling=10, tabOfScenesFirstFramesIndexes=test2)
for key, value in dataPackage.items():
    print(key, value)
"""


book = xlwt.Workbook()
sheet1 = book.add_sheet('pizza')

i = 0
for percent in range(10, 101, 10):
    scenes, dataPackage = main(videoPath, sampling=percent, tabOfScenesFirstFramesIndexes=pizza)
    if percent == 10:
        for key, value in dataPackage.items():
            if i < 5:
                sheet1.write(i, 0, key)
                sheet1.write(i, 1, value)
            else:
                break
            i += 1
    i += 1
    sheet1.write(i, 0, 'sample size (%)')
    sheet1.write(i, 1, dataPackage['sample size (%)'])
    i += 1
    for j in range(dataPackage['number of scenes']):
        sheet1.write(i, j+1, 's' + str(j+1))
    i += 1
    k = 0
    for key, value in dataPackage.items():
        if k > 5:
            sheet1.write(i, 0, key)
            for o, x in enumerate(value):
                sheet1.write(i, o+1, x)
            i += 1
        k += 1

name = "test.xls"
book.save(name)
book.save(TemporaryFile())

