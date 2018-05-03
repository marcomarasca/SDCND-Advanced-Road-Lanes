import pickle
import cv2
import numpy as np
import glob
import os
import argparse
import camera_calibration as cc

class ImageProcessor():

    def __init__(self, calibration_data_file):
        calibration_data = cc.load_calibration_data(file_path = calibration_data_file)
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

    def unwarp_image(self, img):

        img_shape = img.shape[1::-1]

        x = img_shape[0] # 1280
        y = img_shape[1] # 720

        src_x_top_pct = 0.445
        src_y_top_pct = 0.645

        src_x_bot_pct = 0.85
        src_y_bot_pct = 0

        # top left, top right, bottom right, bottom left
        src = np.float32([[x * src_x_top_pct, y * src_y_top_pct], 
                          [x * (1 - src_x_top_pct), y * src_y_top_pct],
                          [x * src_x_bot_pct, img_shape[1] * (1-src_y_bot_pct)], 
                          [x * (1 - src_x_bot_pct), img_shape[1] * (1-src_y_bot_pct)]])

        dst_x_pct = 0.25
        dst_y_pct = 0

        # top left, top right, bottom right, bottom left
        dst = np.float32([[x * dst_x_pct, y * dst_y_pct], 
                          [x * (1 - dst_x_pct), y * dst_y_pct], 
                          [x * (1 - dst_x_pct), y * (1 - dst_y_pct)],
                          [x * dst_x_pct, y * (1 - dst_y_pct)]])

        # Given src and dst points, calculate the perspective transform matrix
        trans_m = cv2.getPerspectiveTransform(src, dst)
        trans_inv_m = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, trans_m, img_shape)

        return warped, trans_m, trans_inv_m

    def undistort_image(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def process_image(self, img):
        result = self.undistort_image(img)
        result, _, _ = self.unwarp_image(result)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image processor')

    parser.add_argument(
        '--file_path',
        type=str,
        default=os.path.join('test_images', '*.jpg'),
        help='File pattern for images to process'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output_images',
        help='Folder where to save the processed images'
    )

    parser.add_argument(
        '--calibration_data_file',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Pickle file containing calibration data'
    )

    args = parser.parse_args()

    img_files = glob.glob(args.file_path)

    imgProcessor = ImageProcessor(args.calibration_data_file)

    for img_file in img_files:
        img = cv2.imread(img_file)
        processed_img = imgProcessor.process_image(img)
        out_file = os.path.join(args.output, os.path.split(img_file)[1][:-4] + '_processed.jpg')
        cv2.imwrite(out_file, processed_img)