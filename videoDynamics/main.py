import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def toGrayscaleAndNormalize(frames):
    numOfFrames = len(frames)
    framesInGrayscale = []
    for i in range(numOfFrames):
        frameInGrayscale = color.rgb2gray(frames[i])
        framesInGrayscale.append(frameInGrayscale)
    return framesInGrayscale

def measureDistance(frames):
    numOfFrames = len(frames)
    partial_d = []
    for i in range(numOfFrames - 1):
        partial_d.append(np.mean(np.abs(frames[i] - frames[i + 1])))
    #print(partial_d)
    return np.mean(partial_d)

def dividingIntoScenes(videoPath, sampling, tabOfScenesTimestamps):
    scenes = []
    vidcap = cv2.VideoCapture(videoPath)

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

def main(videoPath, sampling=1.0, tabOfScenesTimestamps=[]):
    if tabOfScenesTimestamps != []:
        scenes, duration = dividingIntoScenes(videoPath, sampling, tabOfScenesTimestamps)
        scenes_d = []
        for scene in scenes:
            frames = toGrayscaleAndNormalize(scene)
            scenes_d.append(measureDistance(frames))
        print("mean internal scenes distance: ", scenes_d)

        transitions = []
        for k, scene in enumerate(scenes):
            if(k == 0):
                transitions.append(scene[-1])
            elif(k == len(scenes) - 1):
                transitions.append(scene[0])
            else:
                transitions.append(scene[-1])
                transitions.append(scene[0])

        for frame in transitions:
            plt.imshow(frame)
            plt.show()

        transitions_d = []
        for k in range(0, len(transitions), 2):
            frames = toGrayscaleAndNormalize([transitions[k], transitions[k+1]])
            transitions_d.append(measureDistance(frames))
        print("distance between individual scenes", transitions_d)
        sps = len(tabOfScenesTimestamps) / duration # scenes per second
        print("scenes per second:", sps)
    else:
        pass #to-do

videoPath = 'test.mp4'
numOfFrames = False # jeśli false to domyślnie wyciągana jest z filmu jedna klatka na sekundę

im1 = cv2.imread("1.png")[...,::-1]
im2 = cv2.imread("2.png")[...,::-1]
im3 = cv2.imread("3.png")[...,::-1]
im4 = cv2.imread("4.png")[...,::-1]

"""
plt.imshow(frames1[0])
plt.show()

frames2 = [im1, im2]
frames3 = [im3, im4]
array = [frames1, frames2, frames3]

for arr in array:
    frames = toGrayscaleAndNormalize(arr)
    d = measureDistance(frames)
    print(d)
"""

main(videoPath, sampling=0.5, tabOfScenesTimestamps=[0.0, 3.1, 8.0, 11.75, 13.65])





