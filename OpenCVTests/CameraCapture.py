import numpy as np
import cv2
import cv2.cv as cv

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
if __name__ == '__main__':
    # Capture the video stream
    cap = cv2.VideoCapture(0)

    # Load the Haar-cascade classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_eyepair_big.xml')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert to greyscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Find the faces
        faces = detect(gray, face_cascade)
        vis = frame.copy()
        draw_rects(vis, faces, (255,0,0))

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = vis[y:y+h, x:x+w]

            eyes = detect(roi_gray.copy(), eye_cascade)
            draw_rects(roi_color, eyes, (0,255,0))

        # Display the resulting frame
        cv2.imshow('frame',vis)
        
        if 0xFF & cv2.waitKey(5) == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()