import numpy as np
import cv2


def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)

    # Perform Canny edge detection
    edges = 255-cv2.Canny(blurred, 70, 135)

    return blurred, edges


def line_detection(edges, w, h):
    blank = np.zeros(shape=[w, h, 3], dtype=np.uint8)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    if lines is None:
        return blank

    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        y2 = int(y0 - 1000 * (a))
        cv2.line(blank, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return blank


def main():
    cap = cv2.VideoCapture(0)
    WIDTH, HEIGHT = int(cap.get(4)), int(cap.get(3))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Image not captured')
            break

        blurred, edges = canny_edge_detection(frame)

        cv2.imshow('Blurred', blurred)
        cv2.imshow('Edges', edges)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()