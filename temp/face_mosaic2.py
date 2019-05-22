# -*- coding: utf-8 -*-

# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import argparse
import imutils
import cv2
import numpy as np
import os.path as osp
import math
from hpd import HPD

RESIZE_RATIO = 1.4 #프레임 작게하기
SKIP_FRAMES = 3 #프레임 건너뛰기

def main(args):
    filename = args["input_file"]

    if filename is None:
        isVideo = False

        # created a *threaded *video stream, allow the camera sensor to warmup,
        # and start the FPS counter
        print("[INFO] sampling THREADED frames from webcam...")
        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()

    else:
        isVideo = True
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Initialize head pose detection
    hpd = HPD(args["landmark_type"], args["landmark_predictor"])

    cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame2', 320, 240)

    while(vs.stopped == False):
        # Capture frame-by-frame
        print('\rframe: %d' % fps._numFrames, end='')
        frame = vs.read()

        h, w, c = frame.shape
        new_h = (int)(h / RESIZE_RATIO)
        new_w = (int)(w / RESIZE_RATIO)
        frame_small = cv2.resize(frame, (new_w, new_h))
        frame_small2 = cv2.flip(frame_small, 1) # 좌우반전: 카메라 거울상
        # frame_small3 = cv2.flip(frame_small, 0) # 상하반전


        if isVideo:

            if frame is None:
                break
            else:
                out.write(frame)

        else:

            if (fps._numFrames % SKIP_FRAMES == 0):
                frameOut, angles, tvec = hpd.processImage(frame_small2)
                if tvec is None:
                    print('\rframe2: %d' % fps._numFrames, end='')
                    print(" There is no face detected\n")

                    fps.update()
                    #count += 1
                    continue

                else:
                    tx, ty, tz = tvec[:, 0]
                    rx, ry, rz = angles

            else:
                pass


            # Display the resulting frame

            cv2.imshow('frame2',frameOut)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

##        count += 1
        fps.update()

    # When everything done, release the capture
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    vs.stop()
    if isVideo: out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor',
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    parser.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
    parser.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
    parser.add_argument("-p", "--print", type=int, default=-1)
    args = vars(parser.parse_args())
    main(args)
