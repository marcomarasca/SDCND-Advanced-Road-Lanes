from img_processor import ImageProcessor
import numpy as np
import argparse
import cv2
import os
import glob

from collections import deque

class Lanetracker():

    def __init__(self, window_width = 90, window_height = 90, margin = 30, smooth_frames = 15):
        self.frames = deque(maxlen = smooth_frames)
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin

    def find_window_centroids(self, img):

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
        r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-self.window_width/2+int(img.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(img.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(img[int(img.shape[0]-(level+1)*self.window_height):int(img.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,img.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,img.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
        
        self.frames.append(window_centroids)

        if not self.frames:
            return np.array(window_centroids)
        else:
            return np.average(self.frames, axis = 0)

    def curvature(self, window_centroids, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
        
        left_fit_x_vals = window_centroids[:,0]
        right_fit_x_vals = window_centroids[:,1]

        fit_y_vals = np.arange(img.shape[0] - (self.window_height / 2), 0, -self.window_height)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(fit_y_vals * ym_per_pix, left_fit_x_vals * xm_per_pix, 2)
        right_fit_cr = np.polyfit(fit_y_vals * ym_per_pix, right_fit_x_vals * xm_per_pix, 2)
        y_eval = np.max(fit_y_vals)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return (left_curverad, right_curverad)

    def draw_lanes(self, img, window_centroids, blend = False, marker_width = 20):

        left_fit_x_vals = window_centroids[:,0]
        right_fit_x_vals = window_centroids[:,1]

        fit_y_vals = np.arange(img.shape[0] - (self.window_height / 2), 0, -self.window_height)

        left_fit = np.polyfit(fit_y_vals, left_fit_x_vals , 2)
        right_fit = np.polyfit(fit_y_vals, right_fit_x_vals, 2)

        y_vals = range(0, img.shape[0])

        left_x_vals = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2]
        right_x_vals = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2]

        left_lane = list(zip(np.concatenate((left_x_vals - marker_width, left_x_vals[::-1] + marker_width), axis = 0),
            np.concatenate((y_vals, y_vals[::-1]), axis = 0)))
        right_lane = list(zip(np.concatenate((right_x_vals - marker_width, right_x_vals[::-1] + marker_width), axis = 0),
            np.concatenate((y_vals, y_vals[::-1]), axis = 0)))

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

    def draw_windows(self, img, window_centroids, blend = False):
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 	
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = self.window_mask(self.window_width, self.window_height, img, window_centroids[level][0],level)
            r_mask = self.window_mask(self.window_width, self.window_height, img, window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green

        if blend:
            out_img = np.array(cv2.merge((processed_img, processed_img, processed_img)), np.uint8)
            out_img = cv2.addWeighted(out_img, 1.0, template, 0.5, 0)
        else:
            out_img = template

        out_img = self.draw_lanes(out_img, window_centroids, blend = True, marker_width = 3)

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
        default=50,
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

    lane_tracker = Lanetracker(window_width=args.w_width, window_height=args.w_height, margin=args.w_margin, smooth_frames=0)

    for img_file in img_files:
        img = cv2.imread(img_file)
        print("Processing image: {}...".format(img_file))
        processed_img = img_processor.process_image(img)
        print('Finding centroids...')
        window_centroids = lane_tracker.find_window_centroids(processed_img)
        curvature = lane_tracker.curvature(window_centroids)
        print('Curvature: {}'.format(curvature))
        processed_img = lane_tracker.draw_windows(processed_img, window_centroids, blend = True)
        
        out_file_prefix = os.path.join(args.output, os.path.split(img_file)[1][:-4])
        cv2.imwrite(out_file_prefix + '_windows.jpg', processed_img)

        lanes_img = lane_tracker.draw_lanes(img, window_centroids)
        lanes_img = img_processor.unwarp_image(lanes_img)

        undistorted_img = img_processor.undistort_image(img)
        undistorted_img = cv2.addWeighted(undistorted_img, 1.0, lanes_img, 1.0, 0)

        cv2.imwrite(out_file_prefix + '_lanes.jpg', undistorted_img)
        
