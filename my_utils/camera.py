import cv2

cap = cv2.VideoCapture(0)

def get_frame():
  _, frame = cap.read()
  #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return frame #cv2.imread(frame)

def release_capture():
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  while True:
    frame = get_frame()
    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if( cv2.waitKey(1) & 0xFF == ord('q')):
      break

  release_capture()