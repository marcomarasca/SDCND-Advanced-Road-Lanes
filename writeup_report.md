**Advanced Lane Finding Project**

In this project we built a pipeline to detect road lanes in images and video frames. The project consisted in the following steps: 

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to output thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels using sliding window and fit a polynomial to find the road lanes
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_input]: ./camera_cal/calibration2.jpg "Image used for calibration"
[cal_corners]: ./output_images/calibration/calibration2_corners.jpg "Detected corners"
[cal_undistorted]: ./output_images/calibration/calibration2_corners_undistorted.jpg "Undistorsion result"

[p_undistorted]: ./output_images/pipeline_undistorted.jpg "Undistorted image"
[p_color_thresh]: ./output_images/pipeline_color.jpg "Color thresholded image"
[p_gradient_thresh]: ./output_images/pipeline_gradient.jpg "Gradient thresholded image"
[p_thresh]: ./output_images/pipeline_thresholded.jpg "Final thresholded image"
[p_src_dst]: ./output_images/pipeline_src_dst.jpg "Source and destination of the warping"
[p_warped]: ./output_images/pipeline_warped.jpg "Thresholded and warped image"
[p_lanes]: ./output_images/pipeline_lanes.jpg "Sliding windows and lanes detection"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration is contained in the [camera_calibration.py](./camera_calibration.py) file. The script accepts a folder in input that 
is expected to contain calibration images (e.g. images of the chessboard).

I start by preparing the "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `_obj_points` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then proceed to computed the coefficients for the camera calibration and distorsion using the openCV function `cv2.calibrateCamera()`. The result is save in a pickle file for later use.

In the following we can see an example of one of the input calibration images:

![alt text][cal_input]

The result of the corners detections from open cv:

![alt text][cal_corners]

And finally the undistorted image using the coefficients computed from the `calibrationCamera()` function:

![alt text][cal_undistorted]

### Pipeline (single images)

The pipeline can be run using the [img_gen.py](./img_gen.py) script for a given folder, the output is an image for each step of the pipeline. The ImageProcessor in the [img_processor.py](./img_processor.py) file takes care of creating a thresholded and warped image, that is then used by the LaneDetector in the [lane_detector.py](./lane_detector.py) file.

#### 1. Provide an example of a distortion-corrected image.

The pipeline starts off with using the camera calibration data computed previously loading the pickle file and applying the undistorsion to the input images:

![alt text][p_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Next step in the pipeline producing a binary image that tries to underline the road lanes only, excluding the rest from the image. I experimented with different color spaces and gradients combinations, this was definitely the most time consuming part of the project as finding a good combination was a lot of trial and error to find a decent combination that would work for most of the images.

In particular I split my thresholding between color and gradient thresholding, the former can be found in the [img_processor.py](./img_processor.py) within the `color_thresh()` function while the latter is performed in the `gradient_thresh()` function.

Analyzing the various color spaces I noticed that using a combination of RGB, HSL, LAB and HSV I could already extract most of the needed information:

* **yellow lane**: A combination of the B channel from the LAB space and the V channel from HSV (to filter the brightness)
* **white lane**: A combination of the Red channel from RGB and the L channel from HSL (to filter out darker spots)
* **generic lane**: A combination of the S channel from HSL and the V channel from HSV gave good results for extracting both lanes

The various channels are using the following thresholds for their values:

* **R**GB: (195, 255)
* H**S**L: (100, 255)
* HS**L**: (195, 255)
* LA**B**: (150, 255)
* HS**V**: (140, 255)

An example of the result is as follows:

![alt text][p_color_thresh]

I then proceeded using various combination of gradient thresholding, in particular I use complex mix of absolute gradient on both the x and y, the direction and the magnitude of the gradient combined with a filter on the V channel of the HSV color space. This provides relatively good results on a vast amount of different situations, providing additional information where the color thresholding was failing. I ended up with the following combination:

((sobel_x_binary & sobel_y_binary) | sobel_dir_binary) & sobel_mag_binary == 1 & v_binary == 1

Basically relying on the magnitude and the v channel from HSV to filter out noise.

* kernel size: 5
* X: (15, 255)
* Y: (25, 255)
* Magnitude: (40, 255)
* Direction: (0.7, 1.3)
* HS**V**: (180, 255)

An example of the result from the gradient thresholding is as follows:

![alt text][p_gradient_thresh]

The two thresholds are then (or) combined into the final thresholded image:

![alt text][p_thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The third step and final for the image processor is to transform the thresholded images into an bird-eye view applying a perspective transformation, I manually selected source points from an image with straight lines to be mapped into a rectangle-like destination so that I could evaluate visually if the lanes were warped correctly.

I then proceeded to compute the slope and intercept for the line to allow a bit more flexibility in selecting the area of interest. The code can be found in the image processor in the `_warp_coordinates()` function, it also takes into account a trimming from the bottom to remove the hood from the picture.

![alt text][p_src_dst]

The perspective transformation is applied to the thresholded image so that it can be fed to the lane detector:

![alt text][p_warped]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The next part of the pipeline is handled by the [lane detector](./lane_detector.py) that takes in input the thresholded and warped image.

The approach I implemented is using sliding windows applying convolutions. The `detect_lanes` function starts off by applying a convolution to the bottom part of the image to detect where the lanes "start"(`estimate_start_centroids()` function), this first "slice" is bigger than the next windows to be sure to find the starting points.

The resulting centroids are then used in for the next levels of the window sliding algorithm as reference. Each image "slice" produces the centroids for the left and right lane and I implemented a simple "estimation" heuristic to decide where to set the centers based on various scenarios for a given level:

* If the centers were found for both lanes good
* If no center was found then try to predict the points using a partial fitting if enough centroids are found (e.g. if we have 4 points already we can probably already start plotting)
* If only one of the centers is found then uses the previous centers distance to estimate the missing one

This part can be found in the `estimate_centroids()` function.

For each centroid a confidence measure is given by the amount of signal found from the convolution in respect to the maximum amount of signal that can be present in a window (`get_conv_center()` function).

Once we have the centroids for the whole frame a failure detection routine is invoked (`detect_failure()` function), in particular it set thresholds for the mean confidence for each line and compares the mean distance computed for the current frame with a threshold first, with the distance measured in the last frame and finally with the average distance computed for the past X frames, where X is a smoothing factor set in the lane detector (e.g. how many frames are used to smooth the result). I also tried different types of sanity checks such as evaluating the slope of the tangent at the top of the fitted curves, the curvature and deviation comparison but the results were not so stable and introduced too many variables, so I decided to keep it simple.

If the detection fails then the previous frame is reused, the result is then appended to a buffer used for smoothing. Note that for each new frame the previous bottom centers are reused to reduce the initial search for the lane starts.

![alt text][p_lanes]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Finally the curvature and the deviations from the centers are computed in the `compute_curvature()` and `compute_deviation()` functions using the polynomial coefficients scaled to meters per pixel. I visually checked how many pixels are transformed into the warped image and used them to do the scaling. I then applied the radius curvature equation using the first and second derivatives of scaled polynomial using the bottom of the image as my reference point.

For the deviation I simply computed the shift between the detected and scaled distance and the size of the image assuming that the camera is set in the center of the car.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The [ImageProcessor](./img_processor.py) implements an `unwarp()` function that simply applies the inverse perspective transformation. the `detect_lanes()` function in the [LaneDetector](./lane_detector.py) returns the left and right polynomial coefficients so that the lanes can be plot using the `draw_lanes()` function, the result is supplied to the `unwarp()` so that the lanes are "remapped" to the original undistorted image.

![alt text][p_lanes]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_processed.mp4). The pipeline works reasonably well on the project video, while it fails for the [challenge video](./challenge_video_processed.mp4) mostly due to too strict thresholds I chose for the color and gradient thresholding.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a challenging project mostly due to the amount of manual trial and error involved. The color and gradient thresholding were the most time consuming part as finding a general solution for all the situation "hard-coding" thresholds is probably not the best approach. The light conditions may change drastically and so profiles may be built for different scenarios as well as a detection mechanism to apply such profiles. The camera shaking is also a problem, a gimble may be an idea or additional feedback from external sensors in order to adjust the parameters at run time. The pipeline as it is I'd say it's quite limited and makes several assumptions, but due to time constraint I decided to stop to a working implementation. Several additional sanity checks may be implemented that could help in having a smoother and more robust result but my guess is again that external feedback should help rather than having a black box that works with images only.
