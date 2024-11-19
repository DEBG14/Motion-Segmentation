# Stauffer - Grimson Background Subtraction
This project aimed to devise a visual monitoring system that passively observes moving objects in a site and learns patterns of activity from those observations. It provides real-time segmentation of moving regions in a video, separating the foreground and the background.

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
