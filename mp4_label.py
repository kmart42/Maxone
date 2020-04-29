import cv2
import os

# globals
OUTPUT_DIR = './vid_labels/'
MAIN_PATH = '/home/kebin/MaxoneData/Test/SmartcoachTest/assessment2/'
VID_PATH = None
VID_DIR = None
CURR_VID = 'source'
VID_TYPE = '.mp4'
avi_resize = (640, 480)
FOR_KEY = 100  # d
BACK_KEY = 97  # a
SAVE_KEY = 32  # space
ESC_KEY = 27  # escape
POS_STATUS = {0: 0, 1: 0, 2: 0}
POS_LIST = {0: 'playing_hand', 1: 'guide_hand', 2: 'hip_position'}
# change keys based on Dec column at http://www.asciitable.com/
shot_num = 1
shot_pos = None
result = 0


def save_frames(num, res):
    for pos_key in POS_STATUS:
        if POS_STATUS[pos_key] == 1:
            save_file = open(VID_DIR + str(num) + '-' + POS_LIST[pos_key] + ".txt", "w+")
            save_file.write(str(res))
            save_file.close()
            print('Saved ' + str(res) + ' for position ' + POS_LIST[pos_key] + ' in video ' + str(num))


# Function to record the movements of any slider
def update_save(att, val):
    global shot_num
    global shot_pos
    global result
    global POS_LIST
    global POS_STATUS
    if att == 1:
        shot_num = val
        print('current shot number is now ', shot_num)
    elif att == 2:
        if val == 1:
            POS_STATUS[0] = 1
            print(POS_LIST[0] + ' flagged')
        else:
            POS_STATUS[0] = 0
            print(POS_LIST[0] + ' not flagged')
    elif att == 3:
        if val == 1:
            POS_STATUS[1] = 1
            print(POS_LIST[1] + ' flagged')
        else:
            POS_STATUS[1] = 0
            print(POS_LIST[1] + ' not flagged')
    elif att == 4:
        if val == 1:
            POS_STATUS[2] = 1
            print(POS_LIST[2] + ' flagged')
        else:
            POS_STATUS[2] = 0
            print(POS_LIST[2] + ' not flagged')
    else:
        result = val
        print('result is ', result)


def load_vid(load_dir):
    global shot_num
    global shot_pos
    global result
    start_frame = 0
    frame_buffer = 5000
    frames = []
    cap = cv2.VideoCapture(load_dir + CURR_VID + VID_TYPE)
    x = 0
    while True:
        print(x)

        x += 1

        ret, frame = cap.read()
        if not ret or x > frame_buffer + start_frame:
            print('Done read')
            break

        if x < start_frame:
            continue

        frame = cv2.resize(frame, (1280, 720))
        frame = frame[:, 160:-160, :]
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, avi_resize)
        frames.append(frame)

    i = 0

    cv2.namedWindow(CURR_VID)

    # Create sliders for user to identify shot number, attribute, and result
    cv2.createTrackbar('Shot Number', CURR_VID, 0, 10, lambda v: update_save(1, v))
    cv2.createTrackbar('Playing Hand', CURR_VID, 0, 1, lambda v: update_save(2, v))
    cv2.createTrackbar('Guide Hand', CURR_VID, 0, 1, lambda v: update_save(3, v))
    cv2.createTrackbar('Hip Position', CURR_VID, 0, 1, lambda v: update_save(4, v))
    cv2.createTrackbar('Result', CURR_VID, 0, 1, lambda v: update_save(5, v))

    # Iterate through each frame as an individual image
    while i < len(frames):
        print(i)

        frame = frames[i]
        cv2.imshow(CURR_VID, frame)
        key = cv2.waitKey(-1)

        if key == BACK_KEY:
            i -= 1

        elif key == FOR_KEY:
            i += 1

        elif key == SAVE_KEY:
            save_frames(shot_num, result)

        elif key == ESC_KEY:
            i = len(frames) + 1
    cv2.destroyWindow(CURR_VID)


for entry in os.scandir(MAIN_PATH):
    if entry.is_dir():
        pref, path, sufx = str(entry).split("'")
        VID_PATH = path + '/'
        VID_DIR = MAIN_PATH + VID_PATH
        if os.path.isfile(VID_DIR + CURR_VID + VID_TYPE):
            print(VID_DIR)
            load_vid(VID_DIR)
