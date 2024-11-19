from BG_SUB import SG_model
import cv2



GMM=SG_model(0.3,0.75,3)
GMM.parameter_init()
succ=1
count=0
cap=cv2.VideoCapture('umcp.mpg')
while succ:
    succ,frame=cap.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fore,back=GMM.fit(grayscale,frame)
    cv2.imwrite(r"D:/fifa 15/test2/f%d.jpg" % count, fore)
    cv2.imwrite(r"D:/fifa 15/test1/b%d.jpg" % count, back)
    count+=1




