import numpy as np
import cv2
from collections import deque

class LaneDetector:

    FAIL_CODES = {
        1: 'Lane distance not within threshold',
        2: 'Lane distance too different from previous'
    }

    def __init__(self, window_width = 40, window_height = 80, margin = 40, smooth_frames = 15):
        self.centroids_buffer = deque(maxlen = smooth_frames)
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.bottom_pct = .75
        self.min_points_fit = 4 # Number of point already found before trying to fit a line when no center is detected
        self.lanes_dist_thresh = (480, 780)
        self.lanes_dist_max_diff = 30
        self.xm = 3.7/710
        self.ym = 3/120

    def find_lane_start(self, img, window):

        if not self.centroids_buffer:
            l_x_start = 0
            l_x_end = int(img.shape[1] / 2)
            r_x_start = int(img.shape[1] / 2)
            r_x_end = img.shape[1]
        else:
            # If a "good" start was found already, limit the search within the previous
            # frame start boundaries
            prev_centroids = np.array(self.centroids_buffer)
            prev_l_centroids = prev_centroids[:,:,0]
            prev_r_centroids = prev_centroids[:,:,1]
            l_x_start = int(max(np.min(prev_l_centroids) - self.margin, 0))
            l_x_end = int(min(np.max(prev_l_centroids) + self.margin, img.shape[1]))
            r_x_start = int(max(np.min(prev_r_centroids) - self.margin, 0))
            r_x_end = int(min(np.max(prev_r_centroids) + self.margin, img.shape[1]))

        y_start = int(img.shape[0] * self.bottom_pct)
        offset = self.window_width / 2

        l_sum = np.sum(img[y_start:, l_x_start:l_x_end], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) + l_x_start - offset 
        r_sum = np.sum(img[y_start:, r_x_start:r_x_end], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) + r_x_start - offset

        return l_center, r_center, int(img.shape[0] - self.window_height / 2)

    def find_window_center(self, img, window, level, prev_center):

        window_top = int(img.shape[0] - (level + 1) * self.window_height)
        window_bottom = int(img.shape[0] - level * self.window_height)

        # Convolve the window into the vertical slice of the image
        image_slice = np.sum(img[window_top:window_bottom, :], axis=0)
        conv_signal = np.convolve(window, image_slice)

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

        l_center = self.find_window_center(img, window, level, prev_l_center)
        r_center = self.find_window_center(img, window, level, prev_r_center)

        center_y = int((img.shape[0] - level * self.window_height) - self.window_height / 2)

        if l_center is None or r_center is None:
            if len(lanes_centroids) > self.min_points_fit:
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

    def detect_lanes(self, img):

        lanes_centroids = []
        window = np.ones(self.window_width)
   
        l_center, r_center, center_y = self.find_lane_start(img, window)

        # Add what we found for the first layer
        lanes_centroids.append((l_center, r_center, center_y))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / self.window_height)):

            l_center, r_center, center_y = self.estimate_centroids(img, window, level, l_center, r_center, lanes_centroids)

            lanes_centroids.append((l_center, r_center, center_y))

        lanes_centroids = np.array(lanes_centroids)
        
        fail_code = self.detect_failure(lanes_centroids)

        if fail_code > 0 and len(self.centroids_buffer) > 0:
            # In case of failure simply reuse the previous computation
            lanes_centroids = self.centroids_buffer[-1]
            #lanes_centroids = np.average(self.centroids_buffer, axis = 0)

        self.centroids_buffer.append(lanes_centroids)

        if len(self.centroids_buffer) > 0:
            lanes_centroids = np.average(self.centroids_buffer, axis = 0)

        left_fit_scaled, right_fit_scaled = self.lanes_fit(lanes_centroids, ym = self.ym, xm = self.xm)

        curvature = self.compute_curvature(left_fit_scaled, right_fit_scaled, np.max(lanes_centroids[:,2]) * self.ym)
        deviation = self.compute_deviation(img, left_fit_scaled, right_fit_scaled)

        return lanes_centroids, curvature, deviation, fail_code

    def detect_failure(self, lanes_centroids):

        # Checks the mean distance between the two lanes
        lanes_distance = self.compute_mean_distance(lanes_centroids[:,0], lanes_centroids[:,1])
        
        if lanes_distance < self.lanes_dist_thresh[0] or lanes_distance > self.lanes_dist_thresh[1]:
            return 1

        # Checks that the distance is not far apart from previous frames
        if len(self.centroids_buffer) > 0:

            prev_centroids = np.mean(self.centroids_buffer, axis = 0)
            prev_centroids_distance = self.compute_mean_distance(prev_centroids[:,0], prev_centroids[:,1])

            if np.absolute(lanes_distance - prev_centroids_distance) > self.lanes_dist_max_diff:
                return 2

        return 0

    def compute_mean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2) / len(x1))

    def lane_fit(self, lanes_centroids, idx = 0, ym = 1, xm = 1):
        fit_y_vals = lanes_centroids[:,2] * ym
        fit_x_vals = lanes_centroids[:,idx] * xm

        fit = np.polyfit(fit_y_vals, fit_x_vals , 2)

        return fit

    def lanes_fit(self, lanes_centroids, ym = 1, xm = 1):

        left_fit = self.lane_fit(lanes_centroids, 0, ym, xm)
        right_fit = self.lane_fit(lanes_centroids, 1, ym, xm)

        return left_fit, right_fit

    def compute_curvature(self, left_fit, right_fit, y_eval):
       
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

        return (left_curverad, right_curverad)

    def compute_deviation(self, img, left_fit, right_fit):
        y_scaled = img.shape[0] * self.ym
        x_scaled = img.shape[1] * self.xm
        
        l_x = left_fit[0] * y_scaled ** 2 + left_fit[1] * y_scaled + left_fit[2]
        r_x = right_fit[0] * y_scaled ** 2 + right_fit[1] * y_scaled + right_fit[2]
        center = (l_x + r_x) / 2.0
        
        return center - x_scaled / 2.0

    def draw_lanes(self, img, lanes_centroids, blend = False, marker_width = 20, inner_marker = False):

        left_fit, right_fit = self.lanes_fit(lanes_centroids)

        y_vals = range(0, img.shape[0])

        left_x_vals = left_fit[0] * y_vals * y_vals + left_fit[1] * y_vals + left_fit[2]
        right_x_vals = right_fit[0] * y_vals * y_vals + right_fit[1] * y_vals + right_fit[2]

        if blend:
            out_img = img
        else:
            out_img = np.zeros_like(img)

        cv2.polylines(out_img, np.int_([list(zip(left_x_vals, y_vals))]), False, (255, 0, 0), marker_width)
        cv2.polylines(out_img, np.int_([list(zip(right_x_vals, y_vals))]), False, (0, 0, 255), marker_width)

        if inner_marker is True:
            offset = marker_width / 2
            inner_x = np.concatenate((left_x_vals + offset, right_x_vals[::-1] - offset), axis = 0)
            inner_y = np.concatenate((y_vals, y_vals[::-1]), axis = 0)
            cv2.fillPoly(out_img, np.int_([list(zip(inner_x, inner_y))]), color = (0, 255, 0))

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
            l_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][0], center_y)
            r_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][1], center_y)
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
        
