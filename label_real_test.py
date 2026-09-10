"""
Click-to-label tool for Real Test Set keypoints.
Shows each image, click to mark the keypoint, saves ddddd.txt next to the image.
Right-click or press 'u' to undo the last click. Press 'q' to quit early.
"""
import os
import cv2
import numpy as np

IMAGE_DIR = r"D:\fido\Worker\data\data\comp_data\Test Data\Real Test Set"
NUM_IMAGES = 25

click_pos = None
confirmed = False


def mouse_callback(event, x, y, flags, param):
    global click_pos, confirmed
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)
        confirmed = False


def draw(base, point, img_index, total):
    canvas = base.copy()
    h, w = canvas.shape[:2]
    label = f"Image {img_index+1}/{total}  |  click to mark keypoint  |  Enter=confirm  u=undo  q=quit"
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    if point is not None:
        x, y = point
        cv2.drawMarker(canvas, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(canvas, (x, y), 10, (0, 255, 0), 2)
        coord_label = f"({x}, {y})"
        cv2.putText(canvas, coord_label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, coord_label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return canvas


def main():
    global click_pos, confirmed

    win = "Real Test Set Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 900)
    cv2.setMouseCallback(win, mouse_callback)

    for i in range(NUM_IMAGES):
        img_path = os.path.join(IMAGE_DIR, f"{i:05d}.png")
        txt_path = os.path.join(IMAGE_DIR, f"{i:05d}.txt")

        if not os.path.exists(img_path):
            print(f"Missing {img_path}, skipping.")
            continue

        # Skip already-labelled images
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                existing = f.read().strip()
            print(f"{i:05d}.txt already exists ({existing}), skipping.")
            continue

        img = cv2.imread(img_path)
        click_pos = None

        while True:
            canvas = draw(img, click_pos, i, NUM_IMAGES)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("Quit early.")
                cv2.destroyAllWindows()
                return

            if key == ord('u'):
                click_pos = None

            if key in (13, 10):  # Enter
                if click_pos is not None:
                    break
                else:
                    print("Click a point first, then press Enter.")

        x, y = click_pos
        with open(txt_path, "w") as f:
            f.write(f"{x} {y}\n")
        print(f"Saved {i:05d}.txt  →  {x} {y}")

    cv2.destroyAllWindows()
    print("All images labelled.")


if __name__ == "__main__":
    main()
