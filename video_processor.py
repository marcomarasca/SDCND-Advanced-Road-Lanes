import argparse
import os
import cv2
import numpy as np

from img_processor import ImageProcessor
from lane_tracker import LaneTracker

from moviepy.editor import VideoFileClip

class VideoProcessor():

    def __init__(self, file_path, output, calibration_data_file):
        self.file_path = file_path
        self.output = output
        self.img_processor = ImageProcessor(args.calibration_data_file)
        self.lane_tracker = LaneTracker(smooth_frames = 15)

    def process_frame(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        undistorted_img, processed_img, warped_img = self.img_processor.process_image(img)
        window_centroids = self.lane_tracker.find_window_centroids(warped_img)

        lanes_img = self.lane_tracker.draw_lanes(undistorted_img, window_centroids)
        lanes_img = self.img_processor.unwarp_image(lanes_img)

        undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(undistorted_img, 1.0, lanes_img, 1.0, 0)
        #window_img = self.lane_tracker.draw_windows(processed_img, window_centroids)
        #window_img = self.img_processor.unwarp_image(window_img)
        #processed_img = np.array(cv2.merge((processed_img, processed_img, processed_img)), np.uint8)
        #return cv2.addWeighted(processed_img, 1.0, window_img, 0.5, 0)

    def process_video(self):
        #input_clip = VideoFileClip(self.file_path).subclip(t_start = 39, t_end = 45)
        input_clip = VideoFileClip(self.file_path)
        output_clip = input_clip.fl_image(self.process_frame)
        output_clip.write_videofile(self.output, audio = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video processor')

    parser.add_argument(
        'file_path',
        type=str,
        help='Video file path'
    )

    parser.add_argument(
        '--calibration_data_file',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Pickle file containing calibration data'
    )

    args = parser.parse_args()

    output = os.path.split(args.file_path)[1][:-4] + '_processed.mp4'

    video_processor = VideoProcessor(args.file_path, output, args.calibration_data_file)
    video_processor.process_video()