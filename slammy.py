import cv2 as cv
import pygame
from pygame.locals import DOUBLEBUF

video_path = "./test.mp4"

cap = cv.VideoCapture(video_path)

res_scale = 0.5
W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * res_scale)
H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * res_scale)

# INIT DISPLAY
pygame.init()
screen = pygame.display.set_mode((W, H), DOUBLEBUF)
surface = pygame.Surface(screen.get_size())

orb = cv.ORB_create()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:  # If the frame was not read correctly
        print("Error: Could not read frame. Exiting.")
        break
    if frame is None or frame.size == 0:
      continue
      
   
    # Resize the frame if it's not None and has content
    if frame is not None and frame.size > 0:
        frame = cv.resize(frame, (W, H))

        keypoints = orb.detect(frame, None)  # Find keypoints

        keypoints, descriptors = orb.compute(frame, keypoints)  # Compute descriptors with ORB

        frame = cv.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

        # pygame
        pygame.surfarray.blit_array(surface, frame.swapaxes(0, 1)[:, :, [0, 1, 2]])
        screen.blit(surface, (0, 0))
        pygame.display.flip()

# Clean up
cap.release()
pygame.quit()