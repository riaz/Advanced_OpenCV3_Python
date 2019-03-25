# **Finding Lane Lines on the Road** 


### Setup

Runs Jupyter Notebook in a Docker container with `udacity/carnd-term1-starter-kit` image from ![Udacity][docker installation].

```
cd <project-directory>
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

---

**Finding Lane Lines on the Road**

The Project #1 for Udacity’s Self-Driving Car Nanodegree (SDCND) is about creating a pipeline that detects lane lines in images. 
We were previded with several image files, to test our pipeline and we have applied the same pipeline to a video example that was also provided to us, since videos are also streams of images. We made use of the helper functions to implement the pipeline which had code-snippets form the quiz that we have completed earlier. 

We make several improvements on the lane detection method and finish the project by making the system robust enough to work on examples that have curved roads and make the example work.

[//]: # (Image References)

![alt text][image0]

---

### My Pipeline

My Pipeline consists of 6 stages:

1. Grayscale
2. Gaussian Blur
3. Canny Edge Detection
4. Region of Interest 
5. Hough Transform 
6. Draw Lines

The screenshots for the pipeline was created using the following:

     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"gray.jpg"),gray)
     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"blur_gray.jpg"),blur_gray)
     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"edges.jpg"),edges)
     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"masked_edges.jpg"),masked_edges)
     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"line_image.jpg"),line_image)
     cv2.imwrite(os.path.join(IMAGE_SAVE_PATH,"lines_image.jpg"),lines_image)
   

### Reflection

### Grayscale

   For very test image example , we convert the image to grayscale, since its much convenient to apply image transformation function on a grayscale image than a rgb image also the cost of operation is higher and its a very common technique to apply image transformation functions to a grayscale image and apply it as a mask to the original image.
   
  ![alt text][image1]

### Gaussian Blur

   We next smooth the gray scale image with a gaussian kernel of size 7 , so that 
   we can remove noises from the image and make the subsequent image processing steps    easier.
   
   ![alt text][image2]
   
### Canny Edge Detection

   We use the canny edge detection to find the edges in the image, since we know that lane lines are well-defined edges. we can hence use canny detection to get edges and also other edges to be pruned by ROI step next.
   
   ![alt text][image3]
   
### Region of Interest

   After the edge detection setep, we have the scene with many edges, but we are only interested in the edges that form the lane hense we use a trapezoid to mask the scene.
   
   ![alt text][image4]
   
### Hough Transformation

   After the canny edge detection step, all we were left were the edges in the image.
   We then applied ROI to concentrate only on line that form the present lane.
   We use Hough Transform to generate these lines exclusively for the present lane.
   
   ![alt text][image5]
   
### Draw Lines

   I was unable to extrapolate the lines, due to technical issues , and this function
   uses the default functionality. 
   
   ![alt text][image6]


### 2. Identify potential shortcomings with your current pipeline

The short-coming of the current pipeline is that I has out of time implemeting the drawLines() function to be able to extrapolatate the lane lines, since I was facing 
technical dificulties working with the docker instance.

Proposed solution:

I was considering a solution where I would sort the points in the lane lines,
based on the sides they appear and given that I have two sets of points belonging to each lane, I would next

    i.  Find the Top, Bottom and Middle Points 
    ii. In order to make sure the lines are same size, we change the topmost(y) to the
        max(left,right) lane
    iii.We then use the PolyLine() function to draw the lines that would even work
        with the challenge example.



### 3. Suggest possible improvements to your pipeline

My selection of the vertices are hard-coded and and assumed the position of the camera
relative to the car. It would be better if the vertices were a function of the
steering action.


[image0]: ./test_images/solidWhiteCurve.jpg "Original Image"
[image1]: ./test_image_output/gray.jpg "Gray Image"
[image2]: ./test_image_output/blur_gray.jpg "Smoothed Image"
[image3]: ./test_image_output/edges.jpg "Edges Image"
[image4]: ./test_image_output/masked_edges.jpg "Masked Image"
[image5]: ./test_image_output/line_image.jpg "HoughLines Image"
[image6]: ./test_image_output/lines_image.jpg "Output Image"

