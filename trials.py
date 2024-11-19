from BG_SUB import SG_model
import cv2

GMM=SG_model(0.008,0.5,3)
print("start")
GMM.parameter_init()
print("initialization complete")
succ=1
count=0
cap=cv2.VideoCapture('umcp.mpg')
while succ:
    succ,frame=cap.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fore,back = GMM.fit(grayscale,frame)
    cv2.imwrite(r"D:/CUT_PASTE/bg subtraction/outputs/fore/f%d.jpg" % count, fore)
    cv2.imwrite(r"D:/CUT_PASTE/bg subtraction/outputs/back/b%d.jpg" % count, back)
    count+=1




