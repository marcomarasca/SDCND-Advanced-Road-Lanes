import pickle
import cv2
import numpy as np
import glob
import os
import argparse
import camera_calibration as cc

class ImageProcessor():

    def __init__(self, calibration_data_file):

        # Camera calibration data
        calibration_data = cc.load_calibration_data(file_path = calibration_data_file)
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

        # Gradient and color thresholding parameters
        self.sobel_kernel = 15
        self.x_thresh = [20, 100] # Sobel x threshold
        self.y_thresh = [35, 100] # Sobel y threshold
        self.r_thresh = [200, 255] # RGB, Red channel threshold
        self.s_thresh = [100, 255] # HSL, S channel threshold
        self.l_thresh = [200, 255] # HSL, L channel threshold

        # Perspective transformation parameters
        self.persp_src_left_line = (-0.66, 840) # Slope and intercept for left line
        self.persp_src_right_line = (0.66, -5) # Slope and intercept for left line
        self.persp_src_top_pct = 0.65 # Percentage from the top
        self.persp_dst_x_pct = 0.25 # Destination offset percent

    def _sobel(self, img, orient = 'x', sobel_kernel = 3):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        return sobel

    def _apply_thresh(self, ch, thresh = [0, 255]):
        result = np.zeros_like(ch)
        result[(ch >= thresh[0]) & (ch <= thresh[1])] = 1
        return result

    def sobel_abs_thresh(self, sobel, thresh=[0,255]):
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = self._apply_thresh(scaled_sobel, thresh)

        return binary_output

    def sobel_mag_thresh(self, sobel_x, sobel_y, thresh=(0, 255)):
        # Calculate the magnitude 
        abs_sobel = np.sqrt(sobel_x **2 + sobel_y **2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
        # Create a binary mask where mag thresholds are met
        binary_output = self._apply_thresh(scaled_sobel, thresh)
        
        return binary_output

    def sobel_dir_thresh(self, sobel_x, sobel_y, thresh=(0, np.pi/2)):
        # Take the absolute value of the x and y gradients
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)
        # Calculate the direction of the gradient 
        abs_grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
        # Create a binary mask where direction thresholds are met
        binary_output = self._apply_thresh(abs_grad_dir, thresh)
       
        return binary_output

    def color_thresh(self, img, r_thresh = [0, 255], s_thresh = [0, 255], l_thresh = [0, 255]):
        r_ch = img[:,:,2]
        r_binary = self._apply_thresh(r_ch, r_thresh)

        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_ch = hls_img[:,:,2]
        s_binary = self._apply_thresh(s_ch, s_thresh)

        l_ch = hls_img[:,:,1]
        l_binary = self._apply_thresh(l_ch, l_thresh)

        result = np.zeros_like(s_ch)
        result[(r_binary == 1) & (s_binary == 1) | (l_binary == 1)] = 1

        return result

    def _unwarp_coordinates(self, img):

        img_shape = img.shape[1::-1]

        src_y_offset = img_shape[1] * self.persp_src_top_pct
        left_slope, left_intercept = self.persp_src_left_line
        right_slope, right_intercept = self.persp_src_right_line
        
        src = np.float32([[(src_y_offset - left_intercept) / left_slope, src_y_offset],
                          [(src_y_offset - right_intercept) / right_slope, src_y_offset],
                          [(img_shape[1] - right_intercept) / right_slope, img_shape[1]],
                          [(img_shape[1] - left_intercept) / left_slope, img_shape[1]]])

        dst_x_offset = img_shape[0] * self.persp_dst_x_pct

        dst = np.float32([[dst_x_offset, 0], 
                          [img_shape[0] - dst_x_offset, 0], 
                          [img_shape[0] - dst_x_offset, img_shape[1]],
                          [dst_x_offset, img_shape[1]]])
        
        return src, dst

    def unwarp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._unwarp_coordinates(img)

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

        sobel_x = self._sobel(result, sobel_kernel = self.sobel_kernel, orient = 'x')
        sobel_y = self._sobel(result, sobel_kernel = self.sobel_kernel, orient = 'y')

        sobel_x_binary = self.sobel_abs_thresh(sobel_x, self.x_thresh)
        sobel_y_binary = self.sobel_abs_thresh(sobel_y, self.y_thresh)

        sobel_binary = np.zeros_like(sobel_x_binary)
        sobel_binary[(sobel_x_binary == 1 ) & (sobel_y_binary == 1)] = 1

        color_binary = self.color_thresh(result, self.r_thresh, self.s_thresh, self.l_thresh)

        result = np.zeros_like(result)
        result[(color_binary == 1) | (sobel_binary == 1)] = 255

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
       
        out_file_prefix = os.path.join(args.output, os.path.split(img_file)[1][:-4])
        cv2.imwrite(out_file_prefix + '_processed.jpg', processed_img)

        src, dst = imgProcessor._unwarp_coordinates(img)
        dst = np.array(dst, np.int32)
        src = np.array(src, np.int32)

        processed_src = imgProcessor.undistort_image(img)
        processed_src = cv2.polylines(processed_src, [src],True,(0,0,255), 2)
        processed_dst = cv2.polylines(processed_img, [dst],True,(0,0,255), 2)
        
        cv2.imwrite(out_file_prefix + '_persp_src.jpg', processed_src)
        cv2.imwrite(out_file_prefix + '_persp_dst.jpg', processed_dst)

        