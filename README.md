# Motion Segmentation using adaptive mixture models
This repository contains code for the project Motion segmentation which is a part of the Machine learning course (ELL 784) at IIT D, taught by  [Sumantra Dutta Roy](sumantra@ee.iitd.ac.in). The aim is to devise a visual monitoring system that passively observes moving objects in a site and learns patterns of activity from those observations. It provides real-time segmentation of moving regions in a video, separating the foreground and the background.

# Methodology:
The motion segmentation is achieved by modeling each pixel of the frame as a mixture of Gaussians which is updated in real-time as the frames go by. Based on the persistence and the variance of each of the Gaussians of the mixture, the model determines which Gaussians may correspond to the background.

Pixel values that do not fit the background distributions are considered foreground until a Gaussian includes them with sufficient, consistent evidence supporting it.

# Parameters involved:
The backgrounding method required the following significant parameters which needed to be adjusted through trial and error for the best output:

* Learning constant (Alpha) = 0.008
* Background Threshold (T) = 0.5
* High variance Value for non-matching pixels = 1000
* Low prior probability value for non-matching pixels = 0.1

# Output snapshot:
The original video and corresponding Foreground and Background videos are uploaded in the Videos folder. As an illustration, the image below shows the output for a particular frame:
![SGBS](https://github.com/user-attachments/assets/94c2d149-f9ab-4933-87b2-1736e54c1481)
The car backing up and the two people taking a stroll are part of the foreground, while the rest of the objects, being stationary, are classified as a part of the background. It can also be noted that when a foreground object becomes stationary for a bit, (like the car does), the parameters are quickly updated for it to be classified as background for those frames. This is an indication of the system's real-time ability to deal robustly with changes in the scenario.


# Reference:
This project was based on the following papers:

* [Adaptive background mixture models for real-time tracking- Chris Stauffer and W.E.L Grimson 1999](http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf)

* [Learning Patterns of Activity Using Real-Time Tracking- Chris Stauffer and W. Eric L. Grimson 2000](https://people.csail.mit.edu/welg/papers/learning2000.pdf)

