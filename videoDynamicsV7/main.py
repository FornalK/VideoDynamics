import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import random
import pandas as pd
from collections import Counter
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

def L2Norm(h1, h2):
    if (len(h1) != len(h2)):
        print("the histogram vectors are of different length")
        raise Exception
    distance = np.uint64(0)
    for i in range(len(h1)):
        distance += np.uint64((h1[i] - h2[i]) ** 2)

    return np.sqrt(distance)

def agrawalMethod(frames):
    histograms = []
    for frame in frames:
        histogram = Counter(frame.flatten())
        normalizedHistogram = []
        for i in range(256):
            if i in histogram.keys():
                normalizedHistogram.append(histogram[i])
            else:
                normalizedHistogram.append(0)
        histograms.append(normalizedHistogram)

    # x = np.arange(0, 256, dtype=int)
    # for hist in histograms:
    #     plt.bar(x, hist)
    #     plt.show()

    numOfFrames = len(frames)
    partial_d = []
    for i in range(numOfFrames - 1):
        partial_d.append(L2Norm(histograms[i], histograms[i + 1]))
    return np.mean(partial_d)

def extractKeyFrames(videoPath, interval, shift = 0.1):
    dataPackage = {}
    keyFrames = []
    scenes = []
    vidcap = cv2.VideoCapture(videoPath)

    numOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = numOfFrames / fps

    if interval > duration:
        print("Error: specified interval is greater than duration of the movie")
        raise Exception

    success, frame = vidcap.read()
    success = True

    k = 1
    i = shift

    while i < duration:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
        success, frame = vidcap.read()
        if success:
            keyFrames.append(frame[..., ::-1])
        else:
            break
        print("frame num:", k, "\tframe timestamp:", i)
        i += interval
        k += 1

    dataPackage['number of frames'] = numOfFrames
    dataPackage['duration'] = duration
    dataPackage['fps'] = fps
    dataPackage['number of scenes'] = 1
    dataPackage['scenes per second'] = 1 / duration
    dataPackage['interval'] = interval
    dataPackage['number of frames in each scene'] = numOfFrames
    dataPackage['number of frames in sample'] = len(keyFrames)
    dataPackage['number of frames taken'] = len(keyFrames)

    scenes.append(keyFrames)

    return (scenes, duration, dataPackage)

def dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes):
    vidcap = cv2.VideoCapture(videoPath)
    frames = []
    dataPackage = {}
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    scene_num = 0
    sampledScenes = []
    numOfFramesInEachScene = []
    numOfFramesInSample = []
    numberOfFramesTaken = []

    while True:
        if len(tabOfScenesFirstFramesIndexes) > 1:
            if int(pos_frame) == (tabOfScenesFirstFramesIndexes[scene_num + 1]):
                scene_num += 1
                if scene_num == len(tabOfScenesFirstFramesIndexes) - 1:
                    scene_num = 0

                if (sampling > 100 or sampling < 0):
                    raise Exception
                numOfFramesInEachScene.append(len(frames))
                amount = round((sampling / 100) * len(frames))
                if amount == 0:
                    amount = 1
                if amount == 1:
                    middleFrameIndex = len(frames) // 2
                    sampledScene = frames[middleFrameIndex]
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

                    step = round(len(frames) / amount)
                    sampledScene = []
                    for x in range(0, len(frames), step):
                        index = x + step // 2
                        if index < len(frames):
                            sampledScene.append(frames[index])

                    sampledScenes.append(sampledScene)

                    numOfFramesInSample.append(amount)
                    numberOfFramesTaken.append(len(sampledScene))

                frames.clear()

        frame_ready, frame = vidcap.read()
        if frame_ready:
            frames.append(frame[..., ::-1])
            pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            cv2.waitKey(1000)

        if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        if pos_frame % 100 == 0:
            print(pos_frame)

    numOfFramesInEachScene.append(len(frames))
    amount = round((sampling / 100) * len(frames))
    if amount == 0:
        amount = 1
    if amount == 1:
        middleFrameIndex = len(frames) // 2
        sampledScene = frames[middleFrameIndex]
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

        step = round(len(frames) / amount)
        sampledScene = []
        for x in range(0, len(frames), step):
            index = x + step // 2
            if index < len(frames):
                sampledScene.append(frames[index])

        sampledScenes.append(sampledScene)

        numOfFramesInSample.append(amount)
        numberOfFramesTaken.append(len(sampledScene))

    frames.clear()

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    vidcap = None

    dataPackage['number of frames'] = frame_count
    dataPackage['duration'] = duration
    dataPackage['fps'] = fps
    dataPackage['number of scenes'] = len(tabOfScenesFirstFramesIndexes)
    dataPackage['scenes per second'] = len(tabOfScenesFirstFramesIndexes) / duration
    dataPackage['sample size (%)'] = sampling
    dataPackage['number of frames in each scene'] = numOfFramesInEachScene
    dataPackage['number of frames in sample'] = numOfFramesInSample
    dataPackage['number of frames taken'] = numberOfFramesTaken

    return (sampledScenes, duration, dataPackage)

def main(videoPath, sampling = None, interval = None, tabOfScenesFirstFramesIndexes = [], shift = 0.1):
    if sampling == None and interval == None:
        print("Error: only one of the 'interval' or 'sampling' parameters should be specified")
        raise Exception
    if sampling != None and interval != None:
        print("Error: one of the 'interval' or 'sampling' parameters must be specified")
        raise Exception

    if tabOfScenesFirstFramesIndexes != []:
        if sampling != None and interval == None:
            scenes, duration, dataPackage = dividingIntoScenes(videoPath, sampling, tabOfScenesFirstFramesIndexes)
        else: # sampling == None and interval != None:
            if len(tabOfScenesFirstFramesIndexes) != 1:
                print(
                    "Error: If you are extracting frames from a movie with a given interval, treat the video as one scene"
                    "(tabOfScenesFirstFramesIndexes == [0])")
                raise Exception
            else:
                scenes, duration, dataPackage = extractKeyFrames(videoPath, interval, shift)

        internalScenes_d = []
        internalScenes_d2 = []
        print("ilosc scen: ", len(scenes))
        print("liczenie dynamiki wewnętrznej")
        cnt = 0
        for scene in scenes:
            if cnt % 10 == 0:
                print(cnt)
            cnt += 1
            if len(scene) == 1:
                internalScenes_d.append(None)
            else:
                # chinese measure
                frames = toGrayscaleAndNormalize(scene)
                internalScenes_d.append(measureDistance(frames))

                # Prateek Agrawal measure
                internalScenes_d2.append(agrawalMethod(scene))

        print("mean internal scenes distance (chinese): ", internalScenes_d)
        print("mean internal scenes distance (Agrwal): ", internalScenes_d2)

        averagedFrames = []
        for k, scene in enumerate(scenes):
            I, J, K = scene[0].shape
            averagedFrame = np.zeros((I, J, K), dtype=int)
            for frame in scene:
                averagedFrame += frame
            averagedFrame = np.round_(averagedFrame / len(scene)).astype(int)

            averagedFrames.append(averagedFrame)
            # plt.imshow(averagedFrame)
            # plt.show()

        mean_color_values = []
        for frame in averagedFrames:
            temp = frame.copy()
            R = temp[:, :, 0].astype(int)
            G = temp[:, :, 1].astype(int)
            B = temp[:, :, 2].astype(int)
            frameInGrayscale = np.round_((R + G + B) / 3).astype(int)
            mean_color_value = np.mean(frameInGrayscale)
            mean_color_values.append(mean_color_value)

        print("mean_color:", mean_color_values)

        externalScenes_d = []
        externalScenes_d2 = []

        for k in range(len(averagedFrames) - 1):
            # chinese measure
            frames = toGrayscaleAndNormalize([averagedFrames[k], averagedFrames[k+1]])
            externalScenes_d.append(measureDistance(frames))

            # Prateek Agrawal measure
            frames = [averagedFrames[k], averagedFrames[k+1]]
            externalScenes_d2.append(agrawalMethod(frames))

        print("distance between individual scenes (chinese):", externalScenes_d)
        print("distance between individual scenes (Agrawal):", externalScenes_d2)

        """
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
        dataPackage['mean color'] = mean_color_values

        return scenes, dataPackage
    else:
        pass #to-do


# im1 = cv2.imread("f1.png")[...,::-1]
# im2 = cv2.imread("f2.png")[...,::-1]
# im3 = cv2.imread("f3.png")[...,::-1]
# im4 = cv2.imread("4.png")[...,::-1]
# white = cv2.imread("white.png")[...,::-1]
# black = cv2.imread("black.png")[...,::-1]
# gray = cv2.imread("gray.png")[...,::-1]
# frames2 = [im1, im2]
# frames3 = [im1, im3]
# frames4 = [black, white]
# array = [frames2, frames3, frames4]
#
# for arr in array:
#     frames = arr
#     d = agrawalMethod(frames)
#     print(d)


"""
scenes, dataPackage = main(videoPath, sampling=10, tabOfScenesFirstFramesIndexes=test2)
for key, value in dataPackage.items():
    print(key, value)
"""

# i = 0
# for percent in range(10, 101, 10):
#     scenes, dataPackage = main(videoPath, sampling=percent, tabOfScenesFirstFramesIndexes=firstframes)
#     if percent == 10:
#         for key, value in dataPackage.items():
#             if i < 5:
#                 sheet1.write(i, 0, key)
#                 sheet1.write(i, 1, value)
#             else:
#                 break
#             i += 1
#     i += 1
#     sheet1.write(i, 0, 'sample size (%)')
#     sheet1.write(i, 1, dataPackage['sample size (%)'])
#     i += 1
#     for j in range(dataPackage['number of scenes']):
#         sheet1.write(i, j+1, 's' + str(j+1))
#     i += 1
#     k = 0
#     for key, value in dataPackage.items():
#         if k > 5:
#             sheet1.write(i, 0, key)
#             for o, x in enumerate(value):
#                 sheet1.write(i, o+1, x)
#             i += 1
#         k += 1

videoPath = '../6s & 15s/6s - mid1.mp4'
firstframes = open("startframes.txt").read().splitlines()
for i in range(0, len(firstframes)):
    firstframes[i] = int(firstframes[i])

book = xlwt.Workbook()
sheet1 = book.add_sheet('temp')
i = 0

interval = 1
shift = 0.5
sampling = None

scenes, dataPackage = main(videoPath, sampling, interval, tabOfScenesFirstFramesIndexes=firstframes, shift=shift)
scenes_duration = []
for x in range(len(firstframes)):
    if x == len(firstframes) - 1:
        scenes_duration.append((dataPackage['number of frames'] - firstframes[x]) / dataPackage['fps'])
    else:
        scenes_duration.append((firstframes[x + 1] - firstframes[x]) / dataPackage['fps'])

for key, value in dataPackage.items():
    if i < 5:
        sheet1.write(i, 0, key)
        sheet1.write(i, 1, value)
    else:
        break
    i += 1
i += 1

if interval == None:
    sheet1.write(i, 0, 'sample size (%)')
    sheet1.write(i, 1, dataPackage['sample size (%)'])
else:
    sheet1.write(i, 0, 'interval')
    sheet1.write(i, 1, dataPackage['interval'])
i += 2
for j in range(dataPackage['number of scenes']):
    sheet1.write(i+j, 0, 's' + str(j+1))
k = 1
for key, value in dataPackage.items():
    if k > 6:
        sheet1.write(i-1, k-6, key)
        for o, x in enumerate(value):
            sheet1.write(i+o, k-6, x)
    k += 1

sheet1.write(i-1, k-6, 'scene duration')
sheet1.write(i-1, k-5, 'num of first frame')
for j, duration in enumerate(scenes_duration):
        sheet1.write(i+j, k-6, str(duration))
        sheet1.write(i+j, k-5, firstframes[j])

name = "result2.xls"
book.save(name)
book.save(TemporaryFile())

