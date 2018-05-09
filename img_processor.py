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
        print("Loading calibration data...")
        calibration_data = cc.load_calibration_data(file_path = calibration_data_file)
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']
        print(self.mtx, self.dist)

        # Gradient and color thresholding parameters
        self.sobel_kernel = 15
        self.x_thresh = [15, 100] # Sobel x threshold
        self.y_thresh = [30, 100] # Sobel y threshold
        self.dir_thresh = [0.7, 1.3] # Sobel dir threshold

        self.r_thresh = [200, 255] # RGB, Red channel threshold
        self.s_thresh = [100, 255] # HSL, S channel threshold
        self.l_thresh = [195, 255] # HSL, L channel threshold
        self.v_thresh = [170, 255] # HSV, V channel threshold

        # Perspective transformation parameters
        # top left, top right = (585, 456), (700, 456)
        # bottom left, bottom right = (297, 658), (1024, 658)
        self.persp_src_left_line = (-0.6989619377, 864.8927335545) # Slope and intercept for left line
        self.persp_src_right_line = (0.6234567901, 19.58024693) # Slope and intercept for right line
        self.persp_src_top_pct = 0.655 # Percentage from the top
        self.persp_dst_x_pct = 0.25 # Destination offset percent

    def _warp_coordinates(self, img):

        cols = img.shape[1]
        rows = img.shape[0]

        src_y_offset = rows * self.persp_src_top_pct
        left_slope, left_intercept = self.persp_src_left_line
        right_slope, right_intercept = self.persp_src_right_line

        # Top left, Top right, Bottom right, Bottom left        
        src = np.float32([[(src_y_offset - left_intercept) / left_slope, src_y_offset],
                          [(src_y_offset - right_intercept) / right_slope, src_y_offset],
                          [(rows - right_intercept) / right_slope, rows],
                          [(rows - left_intercept) / left_slope, rows]])

        dst_x_offset = cols * self.persp_dst_x_pct

        dst = np.float32([[dst_x_offset, 0], 
                          [cols - dst_x_offset, 0], 
                          [cols - dst_x_offset, rows],
                          [dst_x_offset, rows]])
        
        return src, dst

    def _sobel(self, img, orient = 'x', sobel_kernel = 3):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        return sobel

    def _apply_thresh(self, img, thresh = [0, 255]):
        result = np.zeros_like(img)
        result[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return result

    def unwarp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)
        # Given src and dst points, calculate the perspective transform matrix
        warp_m = cv2.getPerspectiveTransform(dst, src)

        unwarped = cv2.warpPerspective(img, warp_m, img_shape, flags = cv2.INTER_LINEAR)

        return unwarped

    def warp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)

        # Given src and dst points, calculate the perspective transform matrix
        warp_m = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, warp_m, img_shape, flags = cv2.INTER_LINEAR)

        return warped

    def undistort_image(self, img):

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

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

    def gradient_thresh(self, img):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobel_x = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'x')
        sobel_y = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'y')

        sobel_x_binary = self.sobel_abs_thresh(sobel_x, self.x_thresh)
        sobel_y_binary = self.sobel_abs_thresh(sobel_y, self.x_thresh)

        sobel_binary = np.zeros_like(sobel_x_binary)
        sobel_binary[(sobel_x_binary == 1) & (sobel_y_binary == 1)] = 1

        return sobel_binary

    def color_thresh(self, img):
        
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        r_ch = img[:,:,2]
        r_binary = self._apply_thresh(r_ch, self.r_thresh)

        l_ch = hls_img[:,:,1]
        l_binary = self._apply_thresh(l_ch, self.l_thresh)

        s_ch = hls_img[:,:,2]
        s_binary = self._apply_thresh(s_ch, self.s_thresh)

        v_ch = hsv_img[:,:,2]
        v_binary = self._apply_thresh(v_ch, self.v_thresh)

        return r_binary, l_binary, s_binary, v_binary

    def threshold_image(self, img):

        g_binary = self.gradient_thresh(img)
        r_binary, l_binary, s_binary, v_binary = self.color_thresh(img)

        result = np.zeros_like(g_binary)
        result[((s_binary == 1) & (g_binary == 1)) | ((r_binary == 1) & (s_binary == 1)) | ((v_binary == 1) & (g_binary == 1)) ] = 255

        return result

    def process_image(self, img):

        undistorted_img = self.undistort_image(img)

        thresholded_image = self.threshold_image(undistorted_img)

        warped_img = self.warp_image(thresholded_image)

        return undistorted_img, thresholded_image, warped_img

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

    img_processor = ImageProcessor(args.calibration_data_file)

    for img_file in img_files:
        img = cv2.imread(img_file)
        undistorted_img, thresholded_img, processed_warped_img = img_processor.process_image(img)
       
        out_file_prefix = os.path.join(args.output, os.path.split(img_file)[1][:-4])
        cv2.imwrite(out_file_prefix + '_processed.jpg', processed_warped_img)

        cv2.imwrite(out_file_prefix + '_undistorted.jpg', undistorted_img)
        cv2.imwrite(out_file_prefix + '_thresholded.jpg', thresholded_img)
        cv2.imwrite(out_file_prefix + '_gradient.jpg', img_processor.gradient_thresh(undistorted_img) * 255)

        src, dst = img_processor._warp_coordinates(img)

        src = np.array(src, np.int32)
        dst = np.array(dst, np.int32)
        
        warped_img = img_processor.warp_image(undistorted_img)

        processed_src = cv2.polylines(undistorted_img, [src], True, (0,0,255), 2)
        processed_dst = cv2.polylines(warped_img, [dst], True, (0,0,255), 2)
        
        cv2.imwrite(out_file_prefix + '_persp_src.jpg', processed_src)
        cv2.imwrite(out_file_prefix + '_persp_dst.jpg', processed_dst)