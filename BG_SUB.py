import cv2
import numpy as np
import math

class SG_model():

    def __init__(self,alpha,T,K):
        self.alpha=alpha
        self.T=T
        self.K=K
        self.prior_prob=None
        self.mu=None
        self.sigma=None
        self.fore=None
        self.back=None

    def norm(self,x,u,s):
        ans=math.exp(-1/2*((x-u)/s)**2)
        c=math.sqrt(abs(s)*2*math.pi)
        return ans/c


    def match(self,pixel,mu,sigma):
        
        d=abs(pixel - mu)
        if d < (2.5 * abs(sigma) ** (1 / 2.0)):
            return True
        else:
            return False
    
    def Kmeans(self,frame,K): 
       
        rows, cols = frame.shape
        points = rows * cols
        r = np.zeros((points, K))

        #initial vaules of means
        mean = [30,130,230]
        temp = [30,130,230]
        itr = 0


        while (True):
            # clustering each pixel of the image to nearest mean
                for k in range(0, rows):
                    for i in range(0, cols):
                        a = frame[k][i]
                        min = (a - mean[0]) ** 2
                        r[i + cols * k][0] = 0
                        M = 0
                        for j in range(1, K):
                            c = (a - mean[j]) ** 2
                            if c < min:
                                min = c
                                M = j

                            r[i + cols * k][j] = 0

                        r[i + cols * k][M] = 1

                p = np.zeros(K)

            # Calculating the mean of the new clusters
                for j in range(K):
                    p[j] = 1
                    for k in range(0, rows):
                        for i in range(0, cols):
                            mean[j] = mean[j] + frame[k][i] * r[i + cols * k][j]
                            p[j] = p[j] + r[i + cols * k][j]
                    mean[j] = mean[j] / p[j]

            # Check if the new cluster mean is converged below a threshold
                sum = 0
                for j in range(K):
                    sum = sum + (temp[j] - mean[j]) ** 2
                    temp[j] = mean[j]
                if sum < 100:
                      break
                itr += 1

        # Calculate the Variances of the new clusters around the means
        sig = np.zeros(K)
        for j in range(K):
            p[j] = 0
            for k in range(0, rows):
                for i in range(0, cols):
                    sig[j] = sig[j] + (frame[k][i] - mean[j]) ** 2 * r[i + cols * k][j]
                    p[j] = p[j] + r[i + cols * k][j]

            sig[j] = sig[j] / p[j]

        return mean, r, sig
        
       
    
    def parameter_init(self):
        cap=cv2.VideoCapture('umcp.mpg')
        success,init_frame = cap.read()
        gray_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        rows,cols=gray_frame.shape
        points=rows*cols
        self.fore=np.zeros((rows,cols,3))
        self.back=np.zeros((rows,cols,3))
        self.mu = np.ones((points,(self.K)))
        self.prior_prob =  1/self.K * np.ones((points,(self.K)))
        self.sigma = 60 * np.ones((points,(self.K)))
        

        u,r,sig = self.Kmeans(gray_frame,self.K)

        for j in range(self.K):
            for k in range(rows):
                for i in range(cols):
                    self.mu[i + cols * k][j] = u[j]
                    self.sigma[i + cols * k][j] = sig[j]
                    self.prior_prob[i + cols * k][j] = (1 / self.K) * (1 - self.alpha) + self.alpha * r[i + cols * k][j]

        print(self.mu)
        print(self.sigma)
        print(self.prior_prob)
        



        
        
       
        
    def fit(self,frame,original):
        rows,cols=frame.shape
        for i in range(rows):
            for j in range(cols):
                    # check whether pixel matches the existing K Gaussian distributions
                    match = -1
                    for k in range(self.K):
                        if self.match(frame[i][j], self.mu[j+cols*i][k], self.sigma[j+cols*i][k]):
                            match = k
                            break
                    if match != -1:
                        mu = self.mu[j+cols*i][k]
                        s = self.sigma[j+cols*i][k]
                        x = frame[i][j]
                        delta = (x - mu).astype(np.float32)
                        rho = self.alpha * self.norm(frame[i][j],mu,s)
                        self.prior_prob[j+cols*i][k] = (1 - self.alpha) * self.prior_prob[j+cols*i][k]
                        self.prior_prob[j+cols*i][match] += self.alpha
                        self.mu[j+cols*i][k] = mu + rho * delta
                        s=(1-rho)*s + rho*(delta*delta)
                        self.sigma[j+cols*i][k] = s
                    # Normalizing the prior_probs
                    sum=np.sum(self.prior_prob[j+cols*i])
                    ratio=[0 for i in range(self.K)]
                    for k in range(self.K):
                        self.prior_prob[j+cols*i][k]=self.prior_prob[j+cols*i][k]/sum
                        ratio[k]=self.prior_prob[j+cols*i][k]/self.sigma[j+cols*i][k]
                    for k in range(self.K):
                        swapped=False
                        for z in range(self.K-k-1):
                            if ratio[z]<ratio[z+1]:
                                ratio[z],ratio[z+1]=ratio[z+1],ratio[z]
                                self.prior_prob[j+cols*i][z],self.prior_prob[j+cols*i][z+1]=self.prior_prob[j+cols*i][z+1],self.prior_prob[j+cols*i][z]
                                self.mu[j+cols*i][z],self.mu[j+cols*i][z+1]=self.mu[j+cols*i][z+1],self.mu[j+cols*i][z]
                                self.sigma[j+cols*i][z],self.sigma[j+cols*i][z+1]=self.sigma[j+cols*i][z+1],self.sigma[j+cols*i][z]
                                swapped=True
                        if swapped==False:
                            break
                        
                    # if none of the K distributions match the current value
                    # the least probable distribution is replaced with a distribution
                    # with current pixel as its mean, an initially high variance and low prior prob
                    if match == -1:
                        self.mu[j+cols*i][self.K-1] = frame[i][j]
                        self.sigma[j+cols*i][self.K-1] = 800
                        self.prior_prob[j + cols*i][self.K-1]= 0.05
        

                    weight = 0
                    for k in range(self.K):
                         weight += self.prior_prob[j+cols*i][k]
                         if weight > self.T:
                              B = k + 1
                              break
                    for k in range(B):
                         if match==-1 or not(self.match(frame[i][j], self.mu[j+cols*i][k], self.sigma[j+cols*i][k])):
                             self.fore[i][j]=original[i][j]
                             self.back[i][j]=[128,128,128]
                             break
                         else:
                             self.back[i][j]=original[i][j]
                             self.fore[i][j]=[255,255,255]
                             
                             

                
return self.fore,self.back
    
