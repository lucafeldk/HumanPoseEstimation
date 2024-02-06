# Import Packages
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from datetime import datetime

# Code based on: https://github.com/nicknochnack/MultiPoseMovenetLightning


class PoseEstimation:
    def __init__(self):
        # model location
        # self.model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
        self.modelPath = "Models"
        self.model = hub.load(self.modelPath)
        self.movenet = self.model.signatures['serving_default']

        

        # edges that connect
        self.edges = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }

        self.confidence_threshold = 0.2
        self.frame_dimension = [192, 256]

    def get_confidence_threshold(self):
        return self.confidence_threshold

    def set_confidence_threshold(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold

    def get_estimation_dimensions(self):
        return [self.frame_height, self.frame_width]

    def set_estimation_dimension(self, frame_height, frame_width):
        self.frame_dimension = [frame_height, frame_width]

    def draw_connections(self, frame, keypoints):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in self.edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > self.confidence_threshold) & (c2 > self.confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


    def draw_keypoints(self, frame, keypoints):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, conf = kp
            if conf > self.confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

    def loop_through_people(self, frame, keypoints_with_scores, confidence_threshold):
        self.confidence_threshold = confidence_threshold
        for person in keypoints_with_scores:
            self.draw_connections(frame, person)
            self.draw_keypoints(frame, person)

    def transform_frame(self, frame, frame_height, frame_width):
        self.set_estimation_dimension(frame_height, frame_width)
        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), self.frame_dimension[0], self.frame_dimension[1])
        return tf.cast(img, dtype=tf.int32)

def release_Videooutput(out, video_file):
    out.release()
    print("Save the recording(PRESS:y/n)")
    if cv2.waitKey(0) & 0xFF != ord("y"):
        os.remove(video_file)
        print("Video not saved")
    else:
        print("Video saved")

def start_estimation():
    # Start Webcam
    is_video = False
    video_file = "Data//unprocessedVideos//Flower_for_Days_7A_Moonboard.mp4"
    cap = cv2.VideoCapture(video_file)
    estimation = PoseEstimation()
    if is_video is False:
        cap = cv2.VideoCapture(0)


    # define codec and create video writer object -> object for saving videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_videopath = f'Data//processedVideos//output_{datetime.now().strftime("%d_%w_%Y_%H_%M")}.avi'
    out = cv2.VideoWriter(save_videopath, fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))
    print((int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_video:
                release_Videooutput(out,save_videopath)
                break
            print("Can't receive frame!")
            break

        # reshape img
        input_image = estimation.transform_frame(frame.copy(), 192, 256)

        # make keypoint detection
        results = estimation.movenet(input_image)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        #Render keypoints
        estimation.loop_through_people(frame, keypoints_with_scores, 0.2)

        # write frame to file and show
        cv2.imshow('MoveNet Lightning', frame)
        #out.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            release_Videooutput(out, save_videopath)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_estimation()
