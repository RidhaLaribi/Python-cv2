import cv2
import numpy


from PyQt5 import QtWidgets,uic
from PyQt5.uic import *
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QLabel,QFileDialog,QMessageBox
import sys


class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ex4Viewer.ui", self)

        self.cim.clicked.connect(self.loadim)
        self.sim2.clicked.connect(self.showim2)
        self.sim3.clicked.connect(self.showim3)
        self.sim4.clicked.connect(self.showim4)
        self.sim5.clicked.connect(self.showim5)
        self.sim6.clicked.connect(self.showim6)
        self.tedit.setReadOnly(True)

        self.h,self.w=None,None
        self.im=None

    # def loadim(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Choose the image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    #     if not file_path:
    #         return
    #
    #     self.im = cv2.imread(file_path)
    #     if self.im is None:
    #         QtWidgets.QMessageBox.critical(self, "Error", "Failed to load the image. Try another file.")
    #         return
    #
    #     self.h, self.w = self.im.shape[:2]
    #     self.tedit.setPlainText("Image has been loaded successfully")
    #     self.tedit.appendPlainText(f"Height = {self.h}, Width = {self.w}\n")

    def loadim(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose the image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return
        self.im=cv2.imread(file_path)
        if self.im is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load the image. Try another file.")
            return
        self.h,self.w=self.im.shape[:2]
        self.tedit.setPlainText("image has been loaded successfully")
        self.tedit.appendPlainText(f"Height = {self.h}, Width = {self.w}\n")
        cv2.imshow("image",self.im)
        cv2.waitKey(0)


    def showim2(self):
        self.im2=cv2.resize(self.im,[self.w//2,self.h//2])
        cv2.imshow("im2",self.im2)
        self.tedit.appendPlainText("im2 resized successfully.\n")

        cv2.waitKey(0)
    def showim3(self):
        im3=self.im[30:150,200:400]
        cv2.imshow("image3",im3)
        self.tedit.appendPlainText("im3 cropped successfully\n")
        cv2.waitKey(0)
    def showim4(self):
        im4=self.im.copy()
        cv2.rectangle(im4,(30,200),(150,400),(255,0,0),2)
        cv2.putText(im4,"My rectangle",(20,200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
        cv2.imshow("image 4 ",im4)

        self.tedit.appendPlainText("im4 rectangle and text added successfully\n")
        cv2.waitKey(0)

    def showim5(self):
        im5=cv2.rotate(self.im,cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("image 5 ",im5)

        self.tedit.appendPlainText("im5 rotated successfully\n")
        cv2.waitKey(0)
    def showim6(self):
        im6 = cv2.GaussianBlur(self.im, (9, 9), 0)
        cv2.imshow("image 6",im6)

        self.tedit.appendPlainText("im6 blurred successfully\n")
        cv2.waitKey(0)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
