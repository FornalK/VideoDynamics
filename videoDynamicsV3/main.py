import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import random
"""
def toGrayscaleAndNormalize(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for i in range(numOfFrames):
        frameInGrayscale = color.rgb2gray(frames[i])
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale
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

def dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes):
    vidcap = cv2.VideoCapture(videoPath)
    scenes = []
    frames = []

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

    numOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    for k in range(len(tabOfScenesFirstFramesIndexes)):
        if k == len(tabOfScenesFirstFramesIndexes) - 1:
            scenes.append(frames[tabOfScenesFirstFramesIndexes[k]:])
        else:
            scenes.append(frames[tabOfScenesFirstFramesIndexes[k]:tabOfScenesFirstFramesIndexes[k+1]])

    sampledScenes = []
    for scene in scenes:
        if(sampling > 100 or sampling < 0):
            raise
        amount = round((sampling / 100) * len(scene))
        step = round(len(scene) / amount)
        print(amount)
        sampledScene = []
        for x in range(0, len(scene), step):
            sampledScene.append(scene[x])

        print(len(sampledScene))
        sampledScenes.append(sampledScene)

    return (sampledScenes, duration)

def main(videoPath, sampling, tabOfScenesFirstFramesIndexes=[]):
    if tabOfScenesFirstFramesIndexes != []:
        scenes, duration = dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes)
        internalScenes_d = []
        for scene in scenes:
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
            plt.imshow(averagedFrame)
            plt.show()

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

        transitions_d = []
        for k in range(0, len(transitions), 2):
            frames = toGrayscaleAndNormalize([transitions[k], transitions[k + 1]])
            transitions_d.append(measureDistance(frames))
        print("transitions", transitions_d)

        sps = len(tabOfScenesFirstFramesIndexes) / duration  # scenes per second
        print("scenes per second:", sps)

        return scenes
    else:
        pass #to-do

videoPath = 'test.mp4'

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

scenes = main(videoPath, sampling=25, tabOfScenesFirstFramesIndexes=[0, 76, 193, 280, 328])

"""
A = np.array([[120, 200, 210], [100, 95, 35], [53, 35, 51]])
B = np.array([[53, 35, 51], [120, 200, 210], [100, 95, 35]])
C = np.array([A, B])
print(A)
print(B)
print(np.mean(C, axis=0, dtype='uint8'))
"""







