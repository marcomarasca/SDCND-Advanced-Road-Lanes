from img_processor import ImageProcessor
import numpy as np
import argparse
import cv2
import os
import glob

from collections import deque

class LaneTracker():

    def __init__(self, window_width = 60, window_height = 90, margin = 40, smooth_frames = 15):
        self.frames = deque(maxlen = smooth_frames)
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin

    def find_lane_start(self, img, window):
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(img[int(img.shape[0] * .75):, :int(img.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - self.window_width/2
        r_sum = np.sum(img[int(img.shape[0] * .75):, int(img.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width/2+int(img.shape[1] / 2)

        return l_center, r_center

    def get_window_center(self, img, level, conv_signal, prev_center):
        offset = self.window_width/2
        # Find the best centroid by using past center as a reference
        min_index = int(max(prev_center + offset - self.margin, 0))
        max_index = int(min(prev_center + offset + self.margin, img.shape[1]))
        center_max = np.argmax(conv_signal[min_index:max_index])

        # Update the center only if there is some signal
        if center_max > 0:
            center = center_max + min_index - offset
        else:
            center = prev_center

        return center

    def find_lanes_centroids(self, img):

        lane_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions
   
        l_center, r_center = self.find_lane_start(img, window)

        # Add what we found for the first layer
        lane_centroids.append((l_center,r_center))
        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / self.window_height)):
            # Convolve the window into the vertical slice of the image
            image_layer = np.sum(img[int(img.shape[0] - (level + 1) * self.window_height):
                                    int(img.shape[0] - level * self.window_height), :], axis=0)
                                    
            conv_signal = np.convolve(window, image_layer)

            l_center = self.get_window_center(img, level, conv_signal, l_center)
            r_center = self.get_window_center(img, level, conv_signal, r_center)

            # Add what we found for that layer
            lane_centroids.append((l_center,r_center))
        
    
        if not self.frames:
            lane_centroids = np.array(lane_centroids)
        else:
            lane_centroids = np.average(self.frames, axis = 0)
        
        self.frames.append(lane_centroids)
        
        return lane_centroids

    def curvature(self, lanes_centroids, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
        
        left_fit_x_vals = lanes_centroids[:,0]
        right_fit_x_vals = lanes_centroids[:,1]

        fit_y_vals = np.arange(img.shape[0] - (self.window_height / 2), 0, -self.window_height)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(fit_y_vals * ym_per_pix, left_fit_x_vals * xm_per_pix, 2)
        right_fit_cr = np.polyfit(fit_y_vals * ym_per_pix, right_fit_x_vals * xm_per_pix, 2)
        y_eval = np.max(fit_y_vals)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return (left_curverad, right_curverad)

    def draw_lanes(self, img, lanes_centroids, blend = False, marker_width = 20):

        left_fit_x_vals = lanes_centroids[:,0]
        right_fit_x_vals = lanes_centroids[:,1]

        fit_y_vals = np.arange(img.shape[0] - (self.window_height / 2), 0, -self.window_height)

        left_fit = np.polyfit(fit_y_vals, left_fit_x_vals , 2)
        right_fit = np.polyfit(fit_y_vals, right_fit_x_vals, 2)

        y_vals = range(0, img.shape[0])

        left_x_vals = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2]
        right_x_vals = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2]

        y_vals = np.concatenate((y_vals, y_vals[::-1]), axis = 0)

        left_lane = list(zip(np.concatenate((left_x_vals - marker_width, left_x_vals[::-1] + marker_width), axis = 0), y_vals))
        right_lane = list(zip(np.concatenate((right_x_vals - marker_width, right_x_vals[::-1] + marker_width), axis = 0), y_vals))

        if blend:
            out_img = img
        else:
            out_img = np.zeros_like(img)

        cv2.fillPoly(out_img, np.int_([left_lane]), (255, 0, 0))
        cv2.fillPoly(out_img, np.int_([right_lane]), (0, 0, 255))
        
        return out_img

    def window_mask(self, width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def draw_windows(self, img, lanes_centroids, blend = False):
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 	
        for level in range(0, len(lanes_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][0],level)
            r_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | (l_mask == 1) ] = 255
            r_points[(r_points == 255) | (r_mask == 1) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green

        if blend:
            out_img = np.array(cv2.merge((processed_img, processed_img, processed_img)), np.uint8)
            out_img = cv2.addWeighted(out_img, 1.0, template, 0.5, 0)
        else:
            out_img = template

        out_img = self.draw_lanes(out_img, lanes_centroids, blend = True, marker_width = 3)

        return out_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane Tracker')

    parser.add_argument(
        '--file_path',
        type=str,
        default=os.path.join('test_images', '*.jpg'),
        help='File pattern for images to process'
    )

    parser.add_argument(
        '--calibration_data_file',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Pickle file containing calibration data'
    )

    parser.add_argument(
        '--w_width',
        type=int,
        default=60,
        help='Sliding window width'
    )

    parser.add_argument(
        '--w_height',
        type=int,
        default=90,
        help='Sliding window height'
    )

    parser.add_argument(
        '--w_margin',
        type=int,
        default=40,
        help='Sliding window margin from center'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output_images',
        help='Folder where to save the processed images'
    )

    args = parser.parse_args()

    img_files = glob.glob(args.file_path)

    img_processor = ImageProcessor(args.calibration_data_file)

    lane_tracker = LaneTracker(window_width=args.w_width, window_height=args.w_height, margin=args.w_margin, smooth_frames=0)

    for img_file in img_files:
        img = cv2.imread(img_file)
        print("Processing image: {}...".format(img_file))
        undistorted_img, _, processed_img = img_processor.process_image(img)
        print('Finding centroids...')
        lanes_centroids = lane_tracker.find_lanes_centroids(processed_img)
        curvature = lane_tracker.curvature(lanes_centroids)
        print('Curvature: {}'.format(curvature))
        processed_img = lane_tracker.draw_windows(processed_img, lanes_centroids, blend = True)
        
        out_file_prefix = os.path.join(args.output, os.path.split(img_file)[1][:-4])
        cv2.imwrite(out_file_prefix + '_windows.jpg', processed_img)

        lanes_img = lane_tracker.draw_lanes(img, lanes_centroids)
        lanes_img = img_processor.unwarp_image(lanes_img)

        undistorted_img = cv2.addWeighted(undistorted_img, 1.0, lanes_img, 1.0, 0)

        cv2.imwrite(out_file_prefix + '_lanes.jpg', undistorted_img)
        
