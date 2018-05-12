import numpy as np
import cv2
from collections import deque

class LaneDetector:

    FAIL_CODES = {
        1: 'Lane distance not within threshold',
        2: 'Lane distance deviates from mean',
        4: 'Low left lane confidence',
        5: 'Low right lane confidence',
        8: 'Lane deviation differs from previous',
        9: 'Low lanes confidence'
    }

    def __init__(self, window_width = 35, window_height = 90, margin = 35, smooth_frames = 15, xm = 3.7/700, ym = 3/110):
        # [(left, right, y)]
        self.centroids_buffer = deque(maxlen = smooth_frames)
        self.last_lanes_distance = None
        self.last_curvature = None
        self.last_deviation = None
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.first_window_height = .75
        self.min_points_fit = 4 # Number of point already found before trying to fit a line when no center is detected
        self.min_confidence = 0.15
        self.dist_thresh = (590, 810) # Lanes distance threshold
        self.max_dist_dev = 50 # Max lanes distance deviation from mean
        self.max_deviation_dev = 0.1 # Max difference between current deviation and previous
        self.xm = xm
        self.ym = ym

    def compute_window_max_signal(self, window, width, height, max_value = 255):
        window_sum = np.sum(np.ones((height, width)) * max_value, axis = 0)
        conv_signal = np.convolve(window, window_sum)
        return np.max(conv_signal)

    def detect_lanes(self, img):

        lanes_centroids = []
        centroids_confidence = []

        window = np.ones(self.window_width)

        max_signal = self.compute_window_max_signal(window, self.window_width, self.window_height)
   
        left_center, left_confidence, right_center, right_confidence, center_y = self.estimate_start_centroids(img, window)

        # Add what we found for the first layer
        lanes_centroids.append((left_center, right_center, center_y))
        centroids_confidence.append((left_confidence, right_confidence))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / self.window_height)):

            left_center, left_confidence, right_center, right_confidence, center_y = self.estimate_centroids(img, window, max_signal, level, left_center, right_center, lanes_centroids)
            
            lanes_centroids.append((left_center, right_center, center_y))
            centroids_confidence.append((left_confidence, right_confidence))

        lanes_centroids = np.array(lanes_centroids)
        centroids_confidence = np.array(centroids_confidence)

        left_fit_scaled, right_fit_scaled = self.lanes_fit(lanes_centroids, ym = self.ym, xm = self.xm)

        curvature = self.compute_curvature(left_fit_scaled, right_fit_scaled, np.max(lanes_centroids[:,2]) * self.ym)
        deviation = self.compute_deviation(img, left_fit_scaled, right_fit_scaled)
        
        fail_code = self.detect_failure(lanes_centroids, centroids_confidence, curvature, deviation)

        lanes_centroids = self.handle_failure(fail_code, lanes_centroids)

        self.centroids_buffer.append(lanes_centroids)

        left_fit_scaled, right_fit_scaled = self.lanes_fit(lanes_centroids, ym = self.ym, xm = self.xm)

        curvature = self.compute_curvature(left_fit_scaled, right_fit_scaled, np.max(lanes_centroids[:,2]) * self.ym)
        deviation = self.compute_deviation(img, left_fit_scaled, right_fit_scaled)

        if len(self.centroids_buffer) > 0: # Only caches the values when we are smoothing between frames
            self.last_lanes_distance = self.compute_mean_distance(lanes_centroids[:,0], lanes_centroids[:,1])
            self.last_curvature = curvature
            self.last_deviation = deviation
       
        if len(self.centroids_buffer) > 0:
            lanes_centroids = np.average(self.centroids_buffer, axis = 0)

        return lanes_centroids, curvature, deviation, fail_code

    def estimate_start_centroids(self, img, window):

        if not self.centroids_buffer:
            left_min_index = 0
            left_max_index = int(img.shape[1] / 2)
            right_min_index = int(img.shape[1] / 2)
            right_max_index = img.shape[1]
        else:
            # If a "good" start was found already, limit the search within the previous
            # frame start boundaries
            prev_centroids = np.array(self.centroids_buffer)
            prev_left_centroids = prev_centroids[:,:,0]
            prev_right_centroids = prev_centroids[:,:,1]
            left_min_index = int(max(np.min(prev_left_centroids) - self.margin, 0))
            left_max_index = int(min(np.max(prev_left_centroids) + self.margin, img.shape[1]))
            right_min_index = int(max(np.min(prev_right_centroids) - self.margin, 0))
            right_max_index = int(min(np.max(prev_right_centroids) + self.margin, img.shape[1]))

        window_top = int(img.shape[0] * self.first_window_height)
        window_y = int(img.shape[0] - self.window_height / 2)
        
        left_sum = np.sum(img[window_top:, left_min_index:left_max_index], axis=0)
        left_signal = np.convolve(window, left_sum)
        left_center, left_confidence = self.get_conv_center(left_signal, left_min_index)
        
        right_sum = np.sum(img[window_top:, right_min_index:right_max_index], axis=0)
        right_signal = np.convolve(window, right_sum)
        right_center, right_confidence = self.get_conv_center(right_signal, right_min_index)

        return left_center, left_confidence, right_center, right_confidence, window_y

    def get_conv_center(self, conv_signal, offset, max_signal = None):

        max_conv_signal = np.max(conv_signal)

        if max_signal is None or max_conv_signal > 0:
            center = np.argmax(conv_signal) + offset - (self.window_width / 2)
            if max_signal is None: # No max signal given, cannot estimate the confidence
                confidence = 1.0
            else:
                confidence = 0.0 if max_conv_signal == 0 else max_conv_signal / max_signal
        else:
            center = None
            confidence = 0.0
        
        return center, confidence

    def find_window_centroid(self, img, conv_signal, max_signal, prev_center):

        offset = self.window_width / 2
        # Find the best center by using past center as a reference
        min_index = int(max(prev_center + offset - self.margin, 0))
        max_index = int(min(prev_center + offset + self.margin, img.shape[1]))

        conv_window = conv_signal[min_index:max_index]

        center, confidence = self.get_conv_center(conv_window, min_index, max_signal)

        return center, confidence

    def fit_point(self, img, fit, y):
        return np.clip(fit[0]*y**2 + fit[1]*y + fit[2], 0, img.shape[1])

    def estimate_centroids(self, img, window, max_signal, level, prev_l_center, prev_r_center, lanes_centroids):
        window_top = int(img.shape[0] - (level + 1) * self.window_height)
        window_bottom = int(img.shape[0] - level * self.window_height)
        center_y = int(window_bottom - self.window_height / 2)

        # Convolve the window into the vertical slice of the image
        window_sum = np.sum(img[window_top:window_bottom, :], axis=0)

        conv_signal = np.convolve(window, window_sum)

        left_center, left_confidence = self.find_window_centroid(img, conv_signal, max_signal, prev_l_center)
        right_center, right_confidence = self.find_window_centroid(img, conv_signal, max_signal, prev_r_center)

        if left_center is None and right_center is None:
            # If no centers were detected but we have enough points
            # we can try to fit the lane already to get an estimated point
            if len(lanes_centroids) > self.min_points_fit:
                left_fit, right_fit = self.lanes_fit(np.array(lanes_centroids))
                left_center = self.fit_point(img, left_fit, center_y)
                right_center = self.fit_point(img, right_fit, center_y)
            # # If there are at least two elements instead we can use the gap from the previous two frames
            # elif len(lanes_centroids) > 1:
            #     prev_center_gap = np.diff(lanes_centroids[-2:], axis = 0)[0]
            #     left_center = prev_l_center + prev_center_gap[0]
            #     right_center = prev_r_center + prev_center_gap[1]
            # Not enough point, simply use the previous centers (e.g. start centers)
            else:
                left_center = prev_l_center
                right_center = prev_r_center
        # If either one is detected we can use the previous distance as an estimation
        elif left_center is None:
            left_center = right_center - (prev_r_center - prev_l_center)
        elif right_center is None:
            right_center = left_center + (prev_r_center - prev_l_center)

        return left_center, left_confidence, right_center, right_confidence, center_y

    def detect_failure(self, lanes_centroids, centroids_confidence, curvature, deviation):

        left_confidence, right_confidence = np.mean(centroids_confidence, axis = 0)

        confidence_fail = 0
        if left_confidence < self.min_confidence:
            confidence_fail += 4
        if right_confidence < self.min_confidence:
            confidence_fail += 5
        
        if confidence_fail > 0:
            return confidence_fail

        # Checks the mean distance between the two lanes
        lanes_distance = self.compute_mean_distance(lanes_centroids[:,0], lanes_centroids[:,1])

        print('Curv: {:.3f}, Dev: {:.3f}, Dist: {:.3f}, Left Confidence: {:.3f}, Right Confidence: {:.3f}'.format(np.mean(curvature), deviation, lanes_distance, left_confidence, right_confidence))
        
        if len(self.centroids_buffer) > 0:
            prev_centroids = np.mean(self.centroids_buffer, axis = 0)
            prev_centroids_distance = self.compute_mean_distance(prev_centroids[:,0], prev_centroids[:,1])
            print('Prev Curv: {:.3f}, Prev Dev: {:.3f}, Mean Dist: {:.3f}'.format(np.mean(self.last_curvature), self.last_deviation, prev_centroids_distance))

        if self.last_deviation is not None and abs(self.last_deviation - deviation) > self.max_deviation_dev:
            print('D:', self.last_deviation, abs(self.last_deviation - deviation))
            return 8

        if lanes_distance < self.dist_thresh[0] or lanes_distance > self.dist_thresh[1]:
            return 1

        # Checks that the distance is not far apart from previous frames
        if len(self.centroids_buffer) > 0:
            
            prev_centroids = np.mean(self.centroids_buffer, axis = 0)
            prev_centroids_distance = self.compute_mean_distance(prev_centroids[:,0], prev_centroids[:,1])

            if np.absolute(lanes_distance - prev_centroids_distance) > self.max_dist_dev:
                return 2

        return 0

    def handle_failure(self, fail_code, lanes_centroid):
        if fail_code == 0: # No error
            return lanes_centroid
        if fail_code == 4: # Left lane confidence is low
            if self.last_lanes_distance is None:
                lanes_distance = np.sum(self.dist_thresh) / 2
            else:
                lanes_distance = self.last_lanes_distance
            # Uses the same centroids of the right lane shifted
            lanes_centroid[:,0] = lanes_centroid[:,1] - lanes_distance
        if fail_code == 5: # Right lane confidence is low
            if self.last_lanes_distance is None:
                lanes_distance = np.sum(self.dist_thresh) / 2
            else:
                lanes_distance = self.last_lanes_distance
            # Uses the same centroids of the left lane shifted
            lanes_centroid[:,1] = lanes_centroid[:,0] + lanes_distance
        if len(self.centroids_buffer) > 0: # If we have frames we return the previous
            return self.centroids_buffer[-1]
            #lanes_centroids = np.average(self.centroids_buffer, axis = 0)

        return lanes_centroid

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

    def draw_lanes(self, img, lanes_centroids, blend = False, marker_width = 20, fill_color = None):

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

        if fill_color is not None:
            offset = marker_width / 2
            inner_x = np.concatenate((left_x_vals + offset, right_x_vals[::-1] - offset), axis = 0)
            inner_y = np.concatenate((y_vals, y_vals[::-1]), axis = 0)
            cv2.fillPoly(out_img, np.int_([list(zip(inner_x, inner_y))]), color = fill_color)

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
        
