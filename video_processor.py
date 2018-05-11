import argparse
import os
import cv2
import numpy as np

from img_processor import ImageProcessor
from lane_detector import LaneDetector

from moviepy.editor import VideoFileClip

class VideoProcessor:

    def __init__(self, calibration_data_file, smooth_frames = 10, debug = False):
        self.img_processor = ImageProcessor(calibration_data_file)
        self.lane_tracker = LaneDetector(smooth_frames = smooth_frames)
        self.count = 17
        self.debug = debug

    def process_frame(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        undistorted_img, thresholded_img, warped_img = self.img_processor.process_image(img)
        
        lanes_centroids, curvature, deviation, fail_code = self.lane_tracker.detect_lanes(warped_img)
        
        lane_img = self.lane_tracker.draw_lanes(undistorted_img, lanes_centroids, inner_marker = True)
        lane_img = self.img_processor.unwarp_image(lane_img)

        out_image = cv2.addWeighted(undistorted_img, 1.0, lane_img, 1.0, 0)

        font = cv2.FONT_HERSHEY_COMPLEX
        font_color = (0, 255, 0)
        
        cv2.putText(out_image, 'Curvature left: {}, Curvature right: {})'.format(curvature[0], curvature[1]), (30, 60), font, 1, font_color, 2)
        cv2.putText(out_image, 'Deviation from center: {:.2f} m'.format(deviation), (30, 90), font, 1, font_color, 2)

        if self.debug and fail_code > 0:
            cv2.putText(out_image, 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code]), (30, 120), font, 1, (0, 0, 255), 2)
            cv2.putText(img, 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code]), (30, 60), font, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join('test_images', 'test' + str(self.count) + '_failed.png'), img)

        self.count += 1

        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

        return out_image

    def process_video(self, file_path, output, t_start = None, t_end = None):

        input_clip = VideoFileClip(file_path)
        
        if t_start is not None:
            input_clip = input_clip.subclip(t_start = t_start, t_end = t_end)
        
        output_clip = input_clip.fl_image(self.process_frame)
        output_clip.write_videofile(output, audio = False)

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

    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        help='Number of frames to smooth'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=None,
        help='Time start'
    )

    parser.add_argument(
        '--end',
        type=float,
        default=None,
        help='Time start'
    )

    args = parser.parse_args()

    output = os.path.split(args.file_path)[1][:-4] + '_processed.mp4'

    video_processor = VideoProcessor(args.calibration_data_file, smooth_frames = args.smooth, debug=True)
    video_processor.process_video(args.file_path, output, t_start = args.start, t_end = args.end)