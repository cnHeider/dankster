import sys

from PyQt5 import QtCore, QtGui, QtWidgets

def show_image(image_path='/home/heider/Pictures/pandas.jpg'):
  app = QtWidgets.QApplication(sys.argv)
  pixmap = QtGui.QPixmap(image_path)
  screen = QtWidgets.QLabel()
  screen.setAlignment(QtCore.Qt.AlignCenter)
  screen.setPixmap(pixmap)
  screen.showFullScreen()
  sys.exit(app.exec_())

if __name__ == '__main__':
  show_image()
