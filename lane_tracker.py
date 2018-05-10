import numpy as np
import cv2
from collections import deque

class LaneTracker:

    def __init__(self, window_width = 35, window_height = 80, margin = 40, smooth_frames = 15):
        self.frames = deque(maxlen = smooth_frames)
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.bottom_pct = .75

    def find_lane_start(self, img, window):
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum bottom of image to get slice, could use a different ratio
        y_start = int(img.shape[0] * self.bottom_pct)
        x_mid = int(img.shape[1] / 2)
        offset = self.window_width / 2

        l_sum = np.sum(img[y_start:, :x_mid], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - offset
        r_sum = np.sum(img[y_start:, x_mid:], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - offset + x_mid

        return l_center, r_center, int(img.shape[0] - self.window_height / 2)

    def get_window_center(self, img, conv_signal, prev_center):
        offset = self.window_width / 2
        # Find the best center by using past center as a reference
        min_index = int(max(prev_center + offset - self.margin, 0))
        max_index = int(min(prev_center + offset + self.margin, img.shape[1]))

        center_max = np.argmax(conv_signal[min_index:max_index])
        
        # Update the center only if there is some signal
        if center_max > 0:
            center = center_max + min_index - offset
        else:
            center = None

        return center

    def fit_point(self, img, fit, y):
        return np.clip(fit[0]*y**2 + fit[1]*y + fit[2], 0, img.shape[1])

    def estimate_centroids(self, img, window, level, prev_l_center, prev_r_center, lanes_centroids):
        window_top = int(img.shape[0] - (level + 1) * self.window_height)
        window_bottom = int(img.shape[0] - level * self.window_height)
        center_y = int(window_bottom - self.window_height / 2)
        # Convolve the window into the vertical slice of the image
        image_layer = np.sum(img[window_top:window_bottom, :], axis=0)
                                
        conv_signal = np.convolve(window, image_layer)

        l_center = self.get_window_center(img, conv_signal, prev_l_center)
        r_center = self.get_window_center(img, conv_signal, prev_r_center)

        if l_center is None or r_center is None:
            if len(lanes_centroids) > 4:
                if l_center is None:
                    left_fit = self.lane_fit(np.array(lanes_centroids), 0)
                    l_center = self.fit_point(img, left_fit, center_y)
                if r_center is None:
                    right_fit = self.lane_fit(np.array(lanes_centroids), 1)
                    r_center = self.fit_point(img, right_fit, center_y)
            elif l_center is not None:
                r_center = l_center + (prev_r_center - prev_l_center)
            elif r_center is not None:
                l_center = r_center - (prev_r_center - prev_l_center)
            else:
                l_center = prev_l_center
                r_center = prev_r_center

        return l_center, r_center, center_y

    def find_lanes_centroids(self, img):

        lanes_centroids = []
        window = np.ones(self.window_width)
   
        l_center, r_center, center_y = self.find_lane_start(img, window)

        # Add what we found for the first layer
        lanes_centroids.append((l_center, r_center, center_y))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / self.window_height)):

            l_center, r_center, center_y = self.estimate_centroids(img, window, level, l_center, r_center, lanes_centroids)

            lanes_centroids.append((l_center, r_center, center_y))
    
        if not self.frames:
            lanes_centroids = np.array(lanes_centroids)
        else:
            lanes_centroids = np.average(self.frames, axis = 0)
        
        self.frames.append(lanes_centroids)
        
        return lanes_centroids

    def lane_fit(self, lanes_centroids, idx = 0, ym = 1, xm = 1):
        fit_y_vals = lanes_centroids[:,2] * ym
        fit_x_vals = lanes_centroids[:,idx] * xm

        fit = np.polyfit(fit_y_vals, fit_x_vals , 2)

        return fit

    def lanes_fit(self, lanes_centroids, ym = 1, xm = 1):

        left_fit = self.lane_fit(lanes_centroids, 0, ym, xm)
        right_fit = self.lane_fit(lanes_centroids, 1, ym, xm)

        return left_fit, right_fit

    def curvature(self, lanes_centroids, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
        # Fit new polynomials to x,y in world space
        left_fit_cr, right_fit_cr = self.lanes_fit(lanes_centroids, ym_per_pix, xm_per_pix)

        y_eval = np.max(lanes_centroids[:,2]) * ym_per_pix
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        return (left_curverad, right_curverad)

    def draw_lanes(self, img, lanes_centroids, blend = False, marker_width = 20):

        left_fit, right_fit = self.lanes_fit(lanes_centroids)

        y_vals = range(0, img.shape[0])

        left_x_vals = left_fit[0] * y_vals * y_vals + left_fit[1] * y_vals + left_fit[2]
        right_x_vals = right_fit[0] * y_vals * y_vals + right_fit[1] * y_vals + right_fit[2]

        if blend:
            out_img = img
        else:
            out_img = np.zeros_like(img)

        cv2.polylines(out_img, np.int_([list(zip(left_x_vals, y_vals))]), False, (255,0,0), marker_width)
        cv2.polylines(out_img, np.int_([list(zip(right_x_vals, y_vals))]), False, (0,0,255), marker_width)

        return out_img

    def window_mask(self, width, height, img_ref, x, y):
        output = np.zeros_like(img_ref)
        output[int(y - height/2):int(y + height/2),max(0,int(x-width/2)):min(int(x+width/2),img_ref.shape[1])] = 1
        return output

    def draw_windows(self, img, lanes_centroids, blend = False):
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 	
        for level in range(0, len(lanes_centroids)):
            # Window_mask is a function to draw window areas
            center_y = lanes_centroids[level][2]
            l_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][0],center_y)
            r_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][1],center_y)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | (l_mask == 1) ] = 255
            r_points[(r_points == 255) | (r_mask == 1) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green

        if blend:
            out_img = np.array(cv2.merge((img, img, img)), np.uint8)
            out_img = cv2.addWeighted(out_img, 1.0, template, 0.5, 0)
        else:
            out_img = template

        out_img = self.draw_lanes(out_img, lanes_centroids, blend = True, marker_width = 3)

        return out_img
        
