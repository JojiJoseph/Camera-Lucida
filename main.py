import cv2
import argparse
import numpy as np
parser = argparse.ArgumentParser(
                    prog = 'Camera Lucida',
                    description = 'Overlay an image on live video of paper for easy tracing')
parser.add_argument("-c","--camera", type=str, default="/dev/video2")
parser.add_argument("-i","--image", type=str, default="lena.png")
args = parser.parse_args()

img_name = args.image
camera_address = args.camera

overlay = cv2.imread(img_name)
cam = cv2.VideoCapture(camera_address)


overlay_points = [
    [0,0],
    [overlay.shape[1], 0],
    [overlay.shape[1], overlay.shape[0]],
    [0, overlay.shape[0]]
]

_, cam_test_image = cam.read()

video_points = [
    [0,0],
    [cam_test_image.shape[1], 0],
    [cam_test_image.shape[1], cam_test_image.shape[0]],
    [0, cam_test_image.shape[0]]
]

active_overlay_point = -1
active_video_point = -1

cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

def overlay_mouse_listener(event, x, y, flags, param):
    global active_overlay_point, overlay_points
    if event == cv2.EVENT_LBUTTONDOWN:
        active_overlay_point = -1
        for i in range(4):
            if (x-overlay_points[i][0])**2 + (y-overlay_points[i][1]) ** 2 < 20 ** 2:
                active_overlay_point = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if active_overlay_point != -1:
            overlay_points[active_overlay_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        active_overlay_point = -1

cv2.setMouseCallback("Overlay", overlay_mouse_listener)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
def video_mouse_listener(event, x, y, flags, param):
    global active_video_point, video_points
    if event == cv2.EVENT_LBUTTONDOWN:
        active_video_point = -1
        for i in range(4):
            if (x-video_points[i][0])**2 + (y-video_points[i][1]) ** 2 < 20 ** 2:
                active_video_point = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if active_video_point != -1:
            video_points[active_video_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        active_video_point = -1

cv2.setMouseCallback("Output", video_mouse_listener)
while True:
    ret, img = cam.read()
    overlay_copy = overlay.copy()
    for x, y in overlay_points:
        cv2.circle(overlay_copy, (x,y), 2, (255,0,0), -1)
    cv2.polylines(overlay_copy, np.array(overlay_points).reshape((1,4,2)),True, (255,0,0), 1)
    cv2.imshow("Overlay", overlay_copy)
    img_copy = img.copy()
    H_overlay_to_image,_ = cv2.findHomography(np.array(overlay_points).astype(float), np.array(video_points).astype(float), cv2.RANSAC, 1.0)
    # cv2.warpPerspective(overlay_copy, img_copy, H_overlay_to_image)
    # cv2.imshow("Video", img_copy)
    # overlay = cv2.resize(overlay, img.shape[::-1][1:])
    # out = cv2.addWeighted(img, 0.5, overlay, 0.5, 1)
    mask = np.zeros(img_copy.shape).astype(np.uint8)
    cv2.fillPoly(mask, np.array(video_points).reshape((1,4,2)), (255,255,255))
    out =  cv2.warpPerspective(overlay_copy, H_overlay_to_image, (img_copy.shape[1], img_copy.shape[0]))
    mask = cv2.bitwise_and(mask, out)
    out = cv2.addWeighted(img_copy, 1, mask, 0.5, 1)
    for x, y in video_points:
        cv2.circle(out, (x,y), 2, (255,0,0), -1)
    cv2.polylines(out, np.array(video_points).reshape((1,4,2)),True, (255,0,0), 1)
    cv2.imshow("Output", out)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
