import cv2
import numpy as np
import math


class SG_model:
    def __init__(self, alpha, T, K):   # alpha - learning rate, T - Background threshold , K - Number of Gaussians
        self.alpha = alpha
        self.T = T
        self.K = K
        self.prior_prob = None
        self.mu = None
        self.sigma = None
        self.fore = None
        self.back = None

    def norm(self, x, u, s):
        """Calculates the Gaussian probability density function."""
        c = math.sqrt(abs(s) * 2 * math.pi)
        return math.exp(-0.5 * ((x - u) / s) ** 2) / c

    def match(self, pixel, mu, sigma):
        """Checks if a pixel matches the distribution."""
        d = abs(pixel - mu)
        return d < (2.5 * abs(sigma) ** 0.5)

    def Kmeans(self, frame, K):
        """Applies k-means clustering to initialize GMM parameters."""
        rows, cols = frame.shape
        points = rows * cols

        # Initialize means and clustering variables
        mean = np.linspace(30, 230, K).tolist()  # Spread initial means
        prev_mean = mean.copy()
        r = np.zeros((points, K))  # Cluster responsibility matrix

        max_iterations = 100  # Limit iterations for stability
        for _ in range(max_iterations):
            # Assign pixels to the nearest cluster mean
            for idx, pixel in enumerate(frame.flatten()):
                distances = [(pixel - m) ** 2 for m in mean]
                cluster = np.argmin(distances)
                r[idx, :] = 0
                r[idx, cluster] = 1

            # Update cluster means
            new_mean = [np.sum(frame.flatten() * r[:, k]) / np.sum(r[:, k]) for k in range(K)]
            if np.allclose(new_mean, prev_mean, atol=1e-3):  # Check convergence
                break
            prev_mean = new_mean.copy()

        # Calculate variances for each cluster
        variances = [np.sum(((frame.flatten() - mean[k]) ** 2) * r[:, k]) / np.sum(r[:, k]) for k in range(K)]
        return new_mean, r, variances

    def parameter_init(self, video_path="umcp.mpg"):
        """Initializes GMM parameters."""
        cap = cv2.VideoCapture(video_path)
        success, init_frame = cap.read()
        if not success:
            raise ValueError("Error reading the video file.")
        
        print("hello")

        gray_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_frame.shape
        points = rows * cols

        # Initialize GMM matrices
        self.fore = np.zeros((rows, cols, 3), dtype=np.uint8)
        self.back = np.zeros((rows, cols, 3), dtype=np.uint8)
        self.mu = np.zeros((points, self.K))
        self.prior_prob = np.ones((points, self.K)) / self.K
        self.sigma = np.full((points, self.K), 60)

        print("k means start")
        # K-means initialization
        u, r, sig = self.Kmeans(gray_frame, self.K)
        for k in range(self.K):
            self.mu[:, k] = u[k]
            self.sigma[:, k] = sig[k]
            self.prior_prob[:, k] = (1 / self.K) * (1 - self.alpha) + self.alpha * r[:, k]

    def fit(self, frame, original):
        """Fits GMM to the current frame and separates foreground and background."""
        rows, cols = frame.shape

        flattened_frame = frame.flatten()
        flattened_original = original.reshape(-1, 3)

        print(flattened_frame.shape)


        for idx, pixel in enumerate(flattened_frame):
            # Check if the pixel matches any of the existing K distributions
            match_idx = -1
            for k in range(self.K):
                if self.match(pixel, self.mu[idx, k], self.sigma[idx, k]):
                    match_idx = k
                    break

            if match_idx != -1:
                # Update matched distribution parameters
                mu, s = self.mu[idx, match_idx], self.sigma[idx, match_idx]
                delta = pixel - mu
                rho = self.alpha * self.norm(pixel, mu, s)

                self.prior_prob[idx, match_idx] = (1 - self.alpha) * self.prior_prob[idx, match_idx] + self.alpha
                self.mu[idx, match_idx] = mu + rho * delta
                self.sigma[idx, match_idx] = (1 - rho) * s + rho * (delta ** 2)
            else:
                # Replace least probable distribution
                self.mu[idx, -1] = pixel
                self.sigma[idx, -1] = 1000
                self.prior_prob[idx, -1] = 0.1

            # Normalize probabilities
            self.prior_prob[idx] /= np.sum(self.prior_prob[idx])

            # Calculate foreground/background separation
            weight_sum = np.cumsum(self.prior_prob[idx])
            B = np.searchsorted(weight_sum, self.T)

            # print("reached here")

            is_foreground = all(not self.match(pixel, self.mu[idx, k], self.sigma[idx, k]) for k in range(B))
            if is_foreground:
                self.fore[idx // cols, idx % cols] = flattened_original[idx]
                self.back[idx // cols, idx % cols] = [128, 128, 128]
            else:
                self.fore[idx // cols, idx % cols] = [255, 255, 255]
                self.back[idx // cols, idx % cols] = flattened_original[idx]

        return self.fore, self.back
