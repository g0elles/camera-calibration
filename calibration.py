import cv2
import numpy
import json
from datetime import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
patw, path = 7, 6
objp = numpy.zeros((patw * path, 3))

# Se prepara objp (los puntos de objeto), así (0,0,0), (1,0,0), ... (6,5,0)
for i in range(patw * path):
    objp[i, :2] = numpy.array([i % patw, i / patw], numpy.float32)
objp_list, imgp_list = [], []

print('Choose target:\n (ChessBoard = 1, SymmetricCircles = 2)')
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

    key = cv2.waitKey(10)
    if key == 0x1b:  # tecla ESC
        break
    elif key == 0x20 and ret == True:  # Si encontró esquinas, con la tecla espacio guardará
        time = datetime.now()
        time = time.strftime('%d-%m-%y-%H%M%f')
        cv2.imwrite('data/images/' + time + '.jpg', image)
        print('Saved!')
        objp_list.append(objp.astype(numpy.float32))  # Por cada saved, agrega a objp_list un objp (42x3)
        if (target == 1):
            imgp_list.append(
                corners)  # Por cada saved, agrega a img_list un array con las cordenandas de cada esquina (42x1). En cada fila de ese array hay un array (1x2), que son las coordenadas
        else:
            imgp_list.append(centers)

        if len(objp_list) == 10:
            break

if len(objp_list) >= 3:
    K = numpy.zeros((3, 3), float)
    dist = numpy.zeros((5, 1), float)
    retcal, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, imgp_list, (image.shape[1], image.shape[2]), None,
                                                        None)
    mtx = K.tolist()
    distor = dist.tolist()


    # calculo del error
    def calculate_error(imgp_list, objp_list, rvecs, tvecs, K, dist):
        total_error = 0
        total_points = 0
        for i in range(len(objp_list)):
            imgp_list2, _ = cv2.projectPoints(objp_list[i], rvecs[i], tvecs[i], K, dist)
            total_error += numpy.sum(numpy.abs(imgp_list[i] - imgp_list2) ** 2)
            total_points += len(objp_list[i])

        Error = numpy.sqrt(total_error / total_points)
        return Error


    Error = calculate_error(imgp_list, objp_list, rvecs, tvecs, K, dist)

    result = {
        "K": mtx,
        'Distortion': distor,
        "Error": Error
    }

    print('K = ¥n', K)
    print('Distcoeff= ¥n', dist)
    print('Error ret =', retcal)
    print('Error calculated = ', Error)
    with open('data/calib.json', 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)
else:
    print('Images are not enough')
cap.release()
cv2.destroyAllWindows()
