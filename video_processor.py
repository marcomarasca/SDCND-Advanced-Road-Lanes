import argparse
import os
import cv2
import numpy as np
import time

from img_processor import ImageProcessor
from lane_detector import LaneDetector

from moviepy.editor import VideoFileClip

class VideoProcessor:

    def __init__(self, calibration_data_file, smooth_frames = 5, debug = False, failed_frames_dir = 'failed_frames'):
        self.img_processor = ImageProcessor(calibration_data_file)
        self.lane_tracker = LaneDetector(smooth_frames = smooth_frames)
        self.frame_count = 0
        self.fail_count = 0
        self.debug = debug
        self.failed_frames_dir = failed_frames_dir
        self.last_frame = None

    def process_frame(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        undistorted_img, thresholded_img, warped_img = self.img_processor.process_image(img)
        
        lanes_centroids, curvature, deviation, fail_code = self.lane_tracker.detect_lanes(warped_img)
        
        fill_color = (0, 255, 0) if fail_code == 0 else (0, 255, 255)

        lane_img = self.lane_tracker.draw_lanes(undistorted_img, lanes_centroids, fill_color = fill_color)
        lane_img = self.img_processor.unwarp_image(lane_img)

        out_image = cv2.addWeighted(undistorted_img, 1.0, lane_img, 1.0, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 0)

        curvature_text = 'Left Curvature: {:.1f}, Right Curvature: {:.1f}'.format(curvature[0], curvature[1])
        offset_text = 'Center Offset: {:.2f} m'.format(deviation)

        cv2.putText(out_image, curvature_text, (30, 60), font, 1, font_color, 2)
        cv2.putText(out_image, offset_text, (30, 90), font, 1, font_color, 2)

        if fail_code > 0:
            self.fail_count += 1
            failed_text = 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code])
            cv2.putText(out_image, failed_text, (30, 120), font, 1, (0, 0, 255), 2)
            if self.debug:
                print(failed_text)
                cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '.png'), img)
                if self.last_frame is not None: # Saves also the previous frame for comparison
                    cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '_prev.png'), self.last_frame)
                if self.fail_count > 50:
                    exit(0)

        self.frame_count += 1
        self.last_frame = img

        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

        return out_image

    def process_video(self, file_path, output, t_start = None, t_end = None):

        input_clip = VideoFileClip(file_path)
        
        if t_start is not None:
            input_clip = input_clip.subclip(t_start = t_start, t_end = t_end)
        
        output_clip = input_clip.fl_image(self.process_frame)
        output_clip.write_videofile(output, audio = False)

        print('Number of failed detection: {}'.format(self.fail_count))

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
        '--failed_frames_dir',
        type=str,
        default='failed_frames',
        help='Directory to save the failed frames when debug is enabled'
    )

    parser.add_argument(
        '--smooth',
        type=int,
        default=5,
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

    if not os.path.isdir(args.failed_frames_dir):
        os.makedirs(args.failed_frames_dir)

    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    output = os.path.split(args.file_path)[1][:-4] + '_processed_' + date_time_str + '.mp4'

    video_processor = VideoProcessor(args.calibration_data_file, smooth_frames = args.smooth, debug = True)
    video_processor.process_video(args.file_path, output, t_start = args.start, t_end = args.end)
