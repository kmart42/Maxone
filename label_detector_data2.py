from DetectorBackend.utils.load_avi import load_relevant_frames_labels
from DetectorBackend.make.detector import crop_to_hoop
from DetectorBackend.utils.load_avi import load_avi
import DetectorBackend.hoop.detector as hoop_detect
import DetectorBackend.make.detector as make_detect
import DetectorBackend.utils.load_avi as load_avi2
import pickle
import numpy as np
import cv2
import os

# Globals
MAX_FRAMES = 999999
start_frame = 0  # edit here for files larger than 5000 frames
avi_resize = (640, 480)
FILE_NAME = 'Thu_Jan_30_13-17-10_2020_data'
EXT = '.avi'
DATA_SET = 'Thu_new/'
OUTPUT_PATH = '/home/kebin/MaxoneData/ShootawayNew/' + DATA_SET
#VID_PATH = '/home/kebin/MaxoneData/Test/'
VID_PATH = '/home/kebin/MaxoneData/ShootawayNew/' + DATA_SET + '/'
OUTPUT_PATH_POS = OUTPUT_PATH + '/Positive/'
OUTPUT_PATH_NEG = OUTPUT_PATH + '/Negative/'
OUTPUT_PATH_LABEL = OUTPUT_PATH + 'Labels/'
OUTPUT_PATH_POS_NEW = OUTPUT_PATH + '/PositiveNew/'
OUTPUT_PATH_NEG_NEW = OUTPUT_PATH + '/NegativeNew/'
FOR_KEY = 100  # d
BACK_KEY = 97  # a
POS_SAVE = 119  # w
NEG_SAVE = 115  # s
LAB_SAVE = 119  # w
LAB_REMOVE = 115  # s

# detectors
hoop_detector = hoop_detect.HoopDetector()
make_detector = make_detect.Detector(path='../make/makenetv1.3.tflite')

# Process global
# 1: indirect pos/neg (capture all negatives)
# 2: direct pos/neg (manually add negatives)
# 3: label
# 4: update pos/neg
# 5: combine pickled lists
# 6: direct pos/neg, with pre-formatted frames (no crop)
# 7: misc
PROC_GLOBAL = 3


def crop_frame(blob, frame):
    blob = [(blob[0][0] * 2, blob[0][1] * 2), blob[1] * 4]
    x_min = np.clip(blob[0][1] - blob[1], 0, None).astype(int)
    y_min = np.clip(blob[0][0] - blob[1], 0, None).astype(int)
    size = int(blob[1] * 2)
    crop = frame[x_min:x_min + size, y_min:y_min + size]
    return crop


def create_frame_crop(frame, hoop):
    crop = crop_to_hoop(frame, hoop)
    crop = cv2.resize(crop, (96, 96))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
    return crop


# GLOBAL PROCESS 1
if PROC_GLOBAL == 1:
    min_frame = 0
    if EXT == '.avi':
        frames, labels, min_frame = load_relevant_frames_labels(VID_PATH, FILE_NAME)
    else:
        frames = np.load(VID_PATH + FILE_NAME + EXT)
        with open(VID_PATH + FILE_NAME + '.labels', 'rb') as f:
            labels = pickle.load(f)

    # create a make detector object
    hoop_det = hoop_detect.HoopDetector()

    # run through frames until 10 frames from score adding every 3 frames to negatives
    # on press of w add last frames to positive lists
    positive_frames = []
    stop = False
    i = 2
    negative_save_rate = 10
    hoop_blob = [(160, 120), 40]
    while not stop:
        if i > len(frames) - 3:
            i = len(frames) - 3

        raw_frame = frames[i]

        # get hoop pose
        hoop_det(raw_frame)

        if hoop_det.hoop_radius is not None:
            hoop_blob = [(hoop_det.hoop_center[0] * 2, hoop_det.hoop_center[1] * 2),
                         hoop_det.hoop_radius * 2 * 2]

        cropped_frames = [
            create_frame_crop(frames[i - 6], hoop_blob),
            create_frame_crop(frames[i - 4], hoop_blob),
            create_frame_crop(frames[i - 2], hoop_blob),
            create_frame_crop(frames[i], hoop_blob)]
        cv2.imshow('label_data', np.hstack(cropped_frames))

        # check if we are within 30 frames of the nearest score
        deltas = [np.abs((i + min_frame) - x) for x in labels]
        if min(deltas) < 30:

            key = cv2.waitKey(-1)
            print(i + min_frame)
            if key == BACK_KEY:
                i -= 1

            elif key == FOR_KEY:
                i += 1

            elif key == POS_SAVE:
                print('Saving as positive sample, ', i + min_frame)
                np.save(OUTPUT_PATH_POS + FILE_NAME + '__' + str(i + min_frame), cropped_frames)

            elif key == NEG_SAVE:
                print('Saving as negative sample, ', i + min_frame)
                np.save(OUTPUT_PATH_NEG + FILE_NAME + '__' + str(i + min_frame), cropped_frames)

        else:
            i += 1
            print(i + min_frame)
            cv2.waitKey(-1)

            if i % negative_save_rate == 0:
                print('SAVING NEGATIVE')
                np.save(OUTPUT_PATH_NEG + FILE_NAME + '__' + str(i + min_frame), cropped_frames)

# GLOBAL PROCESS 2
elif PROC_GLOBAL == 2:
    if EXT == '.avi':
        frames = []
        cap = cv2.VideoCapture(VID_PATH + FILE_NAME + EXT)
        x = 0
        while True:
            print(x)

            x += 1

            ret, frame = cap.read()
            if not ret or x > 5000 + start_frame:
                print('Done read')
                break

            if x < start_frame:
                continue

            if type == '.mp4':
                frame = cv2.resize(frame, (1280, 720))
                frame = frame[:, 160:-160, :]
                frame = cv2.resize(frame, avi_resize)

            frames.append(frame)

    else:
        frames = np.load(VID_PATH + FILE_NAME + '.npy')

    skip_rate = 1
    positive_frames = []
    stop = False
    i = 2
    hoop_blob = [(160, 120), 40]
    while not stop:
        if i > len(frames) - skip_rate * 3:
            i = len(frames) - skip_rate * 3

        raw_frame = frames[i]

        # get hoop pose
        hoop_detector(raw_frame)

        if hoop_detector.hoop_radius is not None:
            hoop_blob = [(hoop_detector.hoop_center), hoop_detector.hoop_radius]

        cropped_frames = [crop_frame(hoop_blob, frames[i + x]) for x in range(-2 - skip_rate, 1 + skip_rate)]
        cropped_frames = [cropped_frames[0], cropped_frames[len(cropped_frames) // 2], cropped_frames[-1]]

        cv2.imshow('label_data', np.hstack(cropped_frames))

        # push into detector
        make = make_detector(raw_frame, hoop_blob, debug=False)
        if make:
            print('----- MAKE DETECTED -----')

        # record ball poses if a is pressed
        key = cv2.waitKey(-1)
        print(i)
        if key == BACK_KEY:
            i -= 1

        elif key == FOR_KEY:
            i += 1

        elif key == POS_SAVE:
            print('Saving as positive sample')
            np.save(OUTPUT_PATH_POS + FILE_NAME + '__' + str(i + start_frame), cropped_frames)

        elif key == NEG_SAVE:
            print('Saving as negative sample')
            np.save(OUTPUT_PATH_NEG + FILE_NAME + '__' + str(i + start_frame), cropped_frames)

# GLOBAL PROCESS 3
elif PROC_GLOBAL == 3:
    if EXT == '.avi':
        frames = load_avi(VID_PATH + FILE_NAME + EXT,
                          start_frame=start_frame,
                          load_size=5000)
    else:
        frames = np.load(VID_PATH + FILE_NAME + EXT)

    positive_frames = []
    stop = False

    i = 0
    while i < len(frames):
        print(i)

        frame = frames[i]
        resized = cv2.resize(frame, avi_resize)
        cv2.imshow(FILE_NAME, resized)
        key = cv2.waitKey(-1)

        if key == BACK_KEY:
            i -= 1

        elif key == FOR_KEY:
            i += 1

        elif key == POS_SAVE:
            print('Add as positive sample')
            positive_frames.append(i + start_frame)
        elif key == NEG_SAVE:
            print('Removing any positives from this frame')
            if i in positive_frames:
                del positive_frames[positive_frames == i + start_frame]
                print('Found and removed')
            print(positive_frames)

    with open(OUTPUT_PATH + FILE_NAME + '.labels', 'wb+') as f:
        pickle.dump(positive_frames, f)

# GLOBAL PROCESS 4
elif PROC_GLOBAL == 4:
    def get_labels(directory, frame_buf):
        new_labels = []
        labeled_frames = {}
        for file in os.listdir(directory):
            if file.endswith(".npy"):
                frame_addr = int(file.split('__')[-1].split('.')[0])
                if frame_addr >= frame_buf:
                    new_labels.append(frame_addr)
                    labeled_frames[frame_addr] = np.load(directory + file)
        return new_labels, labeled_frames


    def save_frames(new_labels, out_dir, curr_frames, curr_count):
        for label in new_labels:
            if (len(curr_frames) + curr_count) > label + 5:
                new_frames = curr_frames[label - curr_count - 3], curr_frames[label - curr_count - 1], \
                             curr_frames[label - curr_count + 1]
                np.save(out_dir + FILE_NAME + '__' + str(label), np.asarray(new_frames))

    # load 5000 frames at a time
    frames_processed = 0
    true_frame_counter = 0
    buffer_size = 5000

    complete = False
    while not complete:
        frames = load_avi2.load_avi(VID_PATH + FILE_NAME + EXT, frames_processed, buffer_size)
        true_frame_counter += len(frames)

        # save positives
        positive_labels, _ = get_labels(OUTPUT_PATH_POS, frames_processed)
        save_frames(positive_labels, OUTPUT_PATH_POS_NEW, frames, frames_processed)

        # save negatives
        negative_labels, _ = get_labels(OUTPUT_PATH_NEG, frames_processed)
        save_frames(negative_labels, OUTPUT_PATH_NEG_NEW, frames, frames_processed)

        complete = len(frames) < buffer_size
        frames_processed += buffer_size

    print('Total frames viewed', true_frame_counter)

# GLOBAL PROCESS 5
elif PROC_GLOBAL == 5:
    def load_labels(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


    positive_frames = []
    for file in os.listdir(OUTPUT_PATH_LABEL):
        positive_frames = positive_frames + load_labels(OUTPUT_PATH_LABEL + file)
        # positive_frames.append(load_labels(OUTPUT_PATH_LABEL + file))

    with open(OUTPUT_PATH_LABEL + FILE_NAME + EXT + 'Final.labels', 'wb+') as f:
        pickle.dump(positive_frames, f)

    # for frame in positive_frames:
     #   print(frame)


# GLOBAL PROCESS 6
elif PROC_GLOBAL == 6:
    if EXT == '.avi':
        frames = []
        cap = cv2.VideoCapture(VID_PATH + FILE_NAME + EXT)
        x = 0
        while True:
            print(x)

            x += 1

            ret, frame = cap.read()
            if not ret or x > 5000 + start_frame:
                print('Done read')
                break

            if x < start_frame:
                continue

            if type == '.mp4':
                frame = cv2.resize(frame, (1280, 720))
                frame = frame[:, 160:-160, :]
                frame = cv2.resize(frame, avi_resize)

            frames.append(frame)

    else:
        frames = np.load(VID_PATH + FILE_NAME + '.npy')

    skip_rate = 1
    positive_frames = []
    stop = False
    i = 2
    hoop_blob = [(160, 120), 40]
    while not stop:
        if i > len(frames) - skip_rate * 3:
            i = len(frames) - skip_rate * 3

        raw_frame = frames[i]

        # get hoop pose
        # hoop_detector(raw_frame)

        # if hoop_detector.hoop_radius is not None:
           # hoop_blob = [(hoop_detector.hoop_center), hoop_detector.hoop_radius]

        # cropped_frames = [crop_frame(hoop_blob, frames[i + x]) for x in range(-2 - skip_rate, 1 + skip_rate)]
        # cropped_frames = [cropped_frames[0], cropped_frames[len(cropped_frames) // 2], cropped_frames[-1]]

        # cv2.imshow('label_data', np.hstack(cropped_frames))

        frame = frames[i]
        resized = cv2.resize(frame, avi_resize)
        cv2.imshow(FILE_NAME, resized)

        # push into detector
        #make = make_detector(raw_frame, hoop_blob, debug=False)
        #if make:
        #    print('----- MAKE DETECTED -----')

        # record ball poses if a is pressed
        key = cv2.waitKey(-1)
        print(i)
        if key == BACK_KEY:
            i -= 1

        elif key == FOR_KEY:
            i += 1

        elif key == POS_SAVE:
            print('Saving as positive sample')
            np.save(OUTPUT_PATH_POS + FILE_NAME + '__' + str(i + start_frame), frames)

        elif key == NEG_SAVE:
            print('Saving as negative sample')
            np.save(OUTPUT_PATH_NEG + FILE_NAME + '__' + str(i + start_frame), frames)


# misc
elif PROC_GLOBAL == 7:
    def load_labels(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    positive_frames = load_labels(OUTPUT_PATH_LABEL + FILE_NAME + '_01.labels')
    for frame in positive_frames:
        print(frame)