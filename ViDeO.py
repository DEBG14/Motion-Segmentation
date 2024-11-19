import cv2
import os

path1 = 'D:/fifa 15/test2/'
path2 = 'D:/fifa 15/test/'
out_path = 'D:/background subtraction/video'
out_video_name1 = 'foreground.mp4'
out_video_name2 = 'background.mp4'

#pre_imgs1=os.listdir(path1)

def extract_integer(filename):
    return int(filename.split('.')[0][1:])

pre_imgs1=sorted(os.listdir(path1), key=extract_integer)
pre_imgs2=sorted(os.listdir(path2), key=extract_integer)

print(pre_imgs1)



fore=[] 
back=[]

for i in pre_imgs1:
            i=path1+i
            fore.append(i)

for i in pre_imgs2:
            i=path2+i
            back.append(i)
            


"""cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

frame = cv2.imread(fore[0])
size = list(frame.shape)
del size[2]
size.reverse()


video = cv2.VideoWriter(out_path, cv2_fourcc, 30, size) #output video name, fourcc, fps, size

for i in range(len(fore)): 
    video.write(cv2.imread(fore[i]))
    print('frame ', i+1, ' of ', len(fore))

video.release()
print('outputed video to ', out_path)"""


import numpy as np
import glob

img1_array=[]
img2_array = []
for filename in fore:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img1_array.append(img)

for filename in back:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img2_array.append(img)


out1 = cv2.VideoWriter('projectf.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
out2 = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
 
for i in range(len(img1_array)):
    out1.write(img1_array[i])
out1.release()

for i in range(len(img2_array)):
    out1.write(img2_array[i])
out1.release()




