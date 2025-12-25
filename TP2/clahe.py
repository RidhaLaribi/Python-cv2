import matplotlib.pyplot as plt
import cv2
import numpy as  np

def histogramequ():
    img=cv2.imread("Images/BoatsColor.bmp",cv2.IMREAD_GRAYSCALE)
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    cdf=hist.cumsum()
    cdfNorm=cdf*float(hist.max())/cdf.max()

    plt.figure()
    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.subplot(245)
    plt.plot(hist)
    plt.plot(cdfNorm,color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixel')

    equImg=cv2.equalizeHist(img)
    equhist=cv2.calcHist([equImg],[0],None,[256],[0,256])
    equcdf=equhist.cumsum()
    print(equcdf,'\n',max(equcdf))
    equNorm=equcdf*float(equhist.max()) / equcdf.max() #normal arry ta max same size
    print('uycv\n',equNorm)

    plt.subplot(242)
    plt.imshow(equImg,cmap='gray')
    plt.subplot(246)
    plt.plot(equhist)
    plt.plot(equNorm,color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixel')

    claheObj=cv2.createCLAHE(clipLimit=5,tileGridSize=(8,8))
    ckaheImg=claheObj.apply(img)
    ckahehist = cv2.calcHist([ckaheImg], [0], None, [256], [0, 256])
    ckahecdf = ckahehist.cumsum()
    ckaheNorm = ckahecdf * float(ckahehist.max()) / ckahecdf.max()  # normal arry ta max same size


    # print(ckahecdf, '\n', max(ckahecdf))
    # print('uycv\n', ckaheNorm)

    plt.subplot(243)
    plt.imshow(ckaheImg, cmap='gray')
    plt.subplot(247)
    plt.plot(ckahehist)
    plt.plot(ckaheNorm, color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixel')

    h,w= img.shape[:2]

    hist=cv2.calcHist(img,[0],None,[256],[0,256])
    imgclah=np.zeros_like(img)
    t=32;cm=40
    xtil,ytil=h//t,w//t
    for tx in range(xtil):
        for ty in range(ytil):
            x1 = tx * t
            x2 = x1 + t
            y1 = ty * t
            y2 = y1 + t

            clip = img[x1:x2,y1:y2]

            chist,_=np.histogram(clip,bins=256,range=[0,256])
            iat=[]

            sum=0
            for i in range(256):
                if chist[i]> cm :
                    sum+=chist[i]-cm
                    chist[i]=cm
                else:
                    iat.append(i)
            sum=sum/len(iat)
            for i in iat:
                chist[i]+=sum
            cdfclah=chist.cumsum()
            cdfclah=(cdfclah-cdfclah.min())*float(255)/(cdfclah.max()-cdfclah.min())
            # cdfclah=cdfclah.astype(np.uint8)
            imgclah[x1:x2,y1:y2]= cdfclah[clip]
    # print("#"*50 , '\n',  cdfclah)
    # print("#"*50 , '\n',  chist.cumsum())
    plt.subplot(244)
    plt.imshow(ckaheImg, cmap='gray')
    plt.subplot(248)





    plt.show()


# print('#'*50)
# c=[1,2,3]
# d=[5,6,7]
# print(c[d])
histogramequ()