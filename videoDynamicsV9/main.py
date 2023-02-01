import cv2
import numpy as np
import bisect
import random
from collections import Counter
import RandomNumberGenerator

def toGrayscaleAndNormalize(frames):
    """
    Converts the list of frames to grayscale and normalizes them
    :param frames: list of key frames
    :return: list of normalized frames in grayscale
    """
    numOfFrames = len(frames)
    modifiedFrames = []
    for k in range(numOfFrames):
        R = frames[k][:,:,0].astype(int)
        G = frames[k][:,:,1].astype(int)
        B = frames[k][:,:,2].astype(int)
        modifiedFrame = np.round_((R + G + B) / 3).astype(int) / 255
        modifiedFrames.append(modifiedFrame)
    return modifiedFrames

def measureDistance(frames):
    """
    Calculates the distance between successive frames using the Manhattan norm.
    Returns the average result for all distances
    :param frames: list of normalized frames in grayscale
    :return: mean distance
    """
    numOfFrames = len(frames)
    partialDistance = []

    for i in range(numOfFrames - 1):
        partialDistance.append(np.mean(np.abs(frames[i] - frames[i + 1])))

    return np.mean(partialDistance)

def L2Norm(h1, h2):
    """
    Computes the L2 norm from two vectors
    :param h1: histogram no 1
    :param h2: histogram no 2
    :return: L2 norm result
    """
    if (len(h1) != len(h2)):
        print("The histogram vectors are of different length")
        raise Exception
    distance = np.uint64(0)
    for i in range(len(h1)):
        distance += np.uint64((h1[i] - h2[i]) ** 2)

    return np.sqrt(distance)

def agrawalMethod(frames):
    """
    Calculates the distance between successive frames
    :param frames: list of key frames
    :return: distance
    """
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

    numOfFrames = len(frames)
    partialDistance = []
    for i in range(numOfFrames - 1):
        partialDistance.append(L2Norm(histograms[i], histograms[i + 1]))

    return np.mean(partialDistance)

def extractKeyFrames(videoPath, interval, shift):
    """
    Loads a video file and extracts frames with a given interval from it
    :param videoPath: path to the video file
    :param interval: the time distance with which the next frame will be determined
    :param shift: offset from the start of the video
    :return: tuple of data (key frames, duration, etc.)
    """
    dataPackage = {}
    keyFrames = []
    vidcap = cv2.VideoCapture(videoPath)

    if not vidcap.read()[0]:
        print("Error: Failed to open the specified file")
        raise Exception

    numOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = numOfFrames / fps

    if interval > duration:
        print("Error: specified interval is greater than duration of the movie")
        raise Exception

    i = shift

    while i < duration:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
        success, frame = vidcap.read()
        if success:
            keyFrames.append(frame[..., ::-1])
        else:
            break

        i += interval

    dataPackage['number of frames'] = numOfFrames
    dataPackage['duration'] = duration
    dataPackage['fps'] = fps
    dataPackage['number of frames taken'] = len(keyFrames)

    return (keyFrames, dataPackage)

def videoToFrames(videoPath):
    """
    Loads a video file and returns it as a list of frames
    :param videoPath: path to the video file
    :return: list of frames and data dictionary
    """
    vidcap = cv2.VideoCapture(videoPath)

    if not vidcap.read()[0]:
        print("Error: Failed to open the specified file")
        raise Exception

    dataPackage = {}
    frames = []
    numOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = numOfFrames / fps

    while True:
        success, frame = vidcap.read()
        if success:
            frames.append(frame[..., ::-1])
            posFrame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, posFrame - 1)
            cv2.waitKey(1000)

        if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == numOfFrames:
            break

    dataPackage['number of frames'] = numOfFrames
    dataPackage['duration'] = duration
    dataPackage['fps'] = fps

    return (frames, dataPackage)

def getRandomNonRepeatingIntegers(low, high, size):
    """ Samples :param size: integer numbers in range of
        [:param low:, :param high:) without replacement
        by maintaining a list of ranges of values that
        are permitted.

        This list of ranges is used to map a random number
        of a contiguous a range (`r_n`) to a permissible
        number `r` (from `ranges`).
    """
    if size >= high - low:
        print("Error: the amount of numbers to be drawn cannot be greater than the range")
        raise Exception
    ranges = [high]
    high_ = high - 1
    while len(ranges) - 1 < size:
        # generate a random number from an ever decreasing
        # contiguous range (which we'll map to the true
        # random number).
        # consider an example with low=0, high=10,
        # part way through this loop with:
        #
        # ranges = [0, 2, 3, 7, 9, 10]
        #
        # r_n :-> r
        #   0 :-> 1
        #   1 :-> 4
        #   2 :-> 5
        #   3 :-> 6
        #   4 :-> 8
        r_n = random.randint(low, high_) #całkowita, oba included
        range_index = bisect.bisect_left(ranges, r_n)
        r = r_n + range_index
        for i in range(range_index, len(ranges)):
            if ranges[i] <= r:
                # as many "gaps" we iterate over, as much
                # is the true random value (`r`) shifted.
                r = r_n + i + 1
            elif ranges[i] > r_n:
                break
        # mark `r` as another "gap" of the original
        # [low, high) range.
        ranges.insert(i, r)
        # Fewer values possible.
        high_ -= 1
    # `ranges` happens to contain the result.
    return ranges[:-1]


def drawKeyFrames(frames, framePercentage):
    """
    Randomly selects keyframes from the list
    :param frames: list of frames
    :param framePercentage: value indicating how many frames from the list should be drawn
    :return: list of keyframes
    """
    keyFrames = []

    amount = round((framePercentage / 100) * len(frames))
    if amount < 2:
        amount = 2

    randomFramesIndexes = getRandomNonRepeatingIntegers(0, len(frames), amount)
    for index in randomFramesIndexes:
        keyFrames.append(frames[index].copy())

    return keyFrames


def designateDynamics(videoPath, framePercentage = 10, repetitions = 1):
    """
    Determines the dynamics of the video in a alternative way
    :param videoPath: path to the video file
    :param framePercentage: value indicating how many frames from the video should be drawn
    :param repetitions: value specifying how many times the dynamics calculation is to be repeated
    :return: data package (dictionary)
    """
    if framePercentage <= 0 or framePercentage > 100:
        print("Error: frame percentage must be between (0, 100]")
        raise Exception
    elif repetitions <= 0:
        print("Error: number of repetitions must be positive")
        raise Exception
    else:
        frames, dataPackage = videoToFrames(videoPath)

        chineseMethodResults = []
        agrawalMethodResults = []

        for i in range(repetitions):
            keyFrames = drawKeyFrames(frames, framePercentage)

            # chinese measure
            selectedFrames = toGrayscaleAndNormalize(keyFrames)
            distance = measureDistance(selectedFrames)

            # Prateek Agrawal measure
            distance2 = agrawalMethod(keyFrames)

            chineseMethodResults.append(distance)
            agrawalMethodResults.append(distance2)

        meanChineseMethodResult = np.mean(chineseMethodResults)
        meanAgrawalMethodResult = np.mean(agrawalMethodResults)

        numberOfFramesTaken = len(keyFrames)

        dataPackage['number of frames taken'] = numberOfFramesTaken
        dataPackage['mean internal scenes distance (chinese)'] = meanChineseMethodResult
        dataPackage['mean internal scenes distance (Agrawal)'] = meanAgrawalMethodResult

        return dataPackage

def designateDynamicsStandard(videoPath, interval = 1, shift = 0.1):
    """
    Determines the dynamics of the video in a standard way
    :param videoPath: path to the video file
    :param interval: the time distance with which the next frame will be determined
    :param shift: offset from the start of the video
    :return: data package (dictionary)
    """
    if interval <= 0:
        print("Error: The interval cannot be less than or equal to 0")
        raise Exception
    if shift <= 0:
        print("Error: The shift cannot be less than or equal to 0")
        raise Exception
    keyFrames, dataPackage = extractKeyFrames(videoPath, interval, shift)

    result = None
    result2 = None

    if len(keyFrames) != 1:
        # chinese measure
        frames = toGrayscaleAndNormalize(keyFrames)
        result = measureDistance(frames)

        # Prateek Agrawal measure
        result2 = agrawalMethod(keyFrames)

    dataPackage['mean internal scenes distance (chinese)'] = result
    dataPackage['mean internal scenes distance (Agrawal)'] = result2

    return dataPackage

def main():
    """
    Main function. Accepts data from the user and, based on them, controls the execution of the program
    """
    videoPath = input("Please enter the path to the video file:\n")
    print("1. Determining keyframes by interval")
    print("2. Determining keyframes randomly")
    print("0. Exit")
    mode = int(input("Select the dynamics evaluation mode:\n"))

    if mode == 1:
        interval = float(input("specify the interval:\n"))
        shift = float(input("specify the shift:\n"))
        results = designateDynamicsStandard(videoPath, interval, shift)

        print()
        print("Startup parameters")
        print("interval: ", interval)
        print("shift: ", shift)
        print()

    elif mode == 2:
        framePercentage = float(input("specify the frame percentage:\n"))
        repetitions = int(input("specify the number of repetitions:\n"))
        results = designateDynamics(videoPath, framePercentage, repetitions)

        print()
        print("startup parameters")
        print("frame percentage (%): ", framePercentage)
        print("number of repetitions: ", repetitions)
        print()

    else:
        exit(0)

    print("Results:")
    for key in results:
        print(key, ": ", results[key])

if __name__ == "__main__":
    main()

