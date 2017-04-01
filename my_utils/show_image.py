import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QWidget, QHBoxLayout,
    QLabel, QApplication)
import os
import random


class ImageShower(QWidget):
    def __init__(self, image_path):
        super(ImageShower, self).__init__()
        self.show_image(image_path)

    def show_image(self, prediction):

        hbox = QHBoxLayout(self)

        randomImagePath = get_random_image_path(prediction)
        print (randomImagePath)
        pixmap = QtGui.QPixmap(randomImagePath)

        lbl = QLabel(self)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.move(200, 200)

        self.setWindowTitle("Prediction")
        self.show()

        QtCore.QTimer.singleShot(3000, self.close)


def get_random_image_path(prediction):

    files = os.listdir(prediction)
    index = random.randrange(0, len(files))
    return prediction + "/" + files[index]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imgShower = ImageShower("snoopdog")
    sys.exit(app.exec_())
