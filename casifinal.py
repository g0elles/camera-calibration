import json
import numpy, cv2
from datetime import datetime

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
patw, path = 7, 6
objp = numpy.zeros((patw * path, 3))

for i in range(patw * path):
    objp[i, :2] = numpy.array([i % patw, i / patw], numpy.float32)
objp_list, imgp_list = [], []

print('Choose target:\n (ChessBoard = 1, SymmetricCircles = 2, AsymmetricCircles = 3)')
target = int(input())

while 1:
    stat, image = cap.read(0)

    if target == 1:
        ret, corners = cv2.findChessboardCorners(image, (patw, path), None)
        cv2.drawChessboardCorners(image, (patw, path), corners, ret)
        cv2.imshow('Find ChessBoard', image)
    if target == 2:
        ret, centers = cv2.findCirclesGrid(image, (patw, path), None)
        cv2.drawChessboardCorners(image, (patw, path), centers, ret)
        cv2.imshow('Find Symmetric Circles', image)
    if target == 3:
        ret, centers = cv2.findCirclesGrid(image, (patw, path), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        cv2.drawChessboardCorners(image, (patw, path), centers, ret)
        cv2.imshow('Find Asymmetric Circles', image)

    key = cv2.waitKey(10)
    if key == 0x1b:  # ESC
        break
    elif key == 0x20 and ret == True:
        time = datetime.now()
        time = time.strftime('%d-%m-%y-%H%M%f')
        cv2.imwrite('data/images/' + time + '.jpg', image)
        print('Saved!')
        objp_list.append(objp.astype(numpy.float32))
        if (target == 1):
            imgp_list.append(corners)
        else:
            imgp_list.append(centers)

        if len(objp_list) == 10:
            break

print(objp_list)
if len(objp_list) >= 3:
    K = numpy.zeros((3, 3), float)
    dist = numpy.zeros((5, 1), float)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, imgp_list, (image.shape[1], image.shape[2]), None, None)
    mtx = K.tolist()
    distor = dist.tolist()

    error = 0
    for i in range(len(objp_list)):
        imgPoints2, _ = cv2.projectPoints(objp_list[i], rvecs[i], tvecs[i], K, dist)
        error += cv2.norm(imgp_list[i], imgPoints2, cv2.NORM_L2) / len(imgPoints2)

    Error = error / len(objp_list)

    result = {
        "K": mtx,
        'Distortion': distor,
        "Error": Error
    }

    print('K = ¥n', K)
    """numpy.savetxt('K.txt', K)"""
    print('Distcoeff= ¥n', dist)
    """numpy.savetxt('distCoef.txt', dist)"""
    print('Error = ', Error)
    with open('data/calib.json', 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)
else:
    print('Images are not enough')
cap.release()
cv2.destroyAllWindows()