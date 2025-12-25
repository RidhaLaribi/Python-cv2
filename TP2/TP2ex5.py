import cv2
import numpy
img= cv2.imread("Images/BoatsColor.bmp")
imglab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
# imglab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
def norm(imglab,ymin,ymax,sel):
    hist = cv2.calcHist([imglab], [0], None, [256], [0, 256])
    xmin=0
    for i in range(0,256):
        if hist[i]>sel:
            xmin=i
        break
    xmax=0
    for j in range(255,-1,-1):
        if hist[j] > sel:
            xmax = j
        break

    alpha=((ymin*xmax)-(ymax*xmin))/(xmax-xmin) if (xmax-xmin)!=0 else (ymin*xmax)/0.01
    beta=(ymax-ymin)/(xmax-xmin) if (xmax-xmin)!=0 else (ymax-ymin)/0.01
    print(alpha,beta)
    l,a,b=cv2.split(imglab)[:3]
    r=cv2.normalize(l,None,int(alpha),int(beta),cv2.NORM_MINMAX)


    img1=cv2.merge([r,a,b])
    im=cv2.cvtColor(img1,cv2.COLOR_Lab2RGB)
    return im

im=norm(imglab,10,20,30)
cv2.imshow("org",img)
cv2.imshow("fj",im)
cv2.waitKey(0)

