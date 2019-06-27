import json
import numpy, cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
patw, path = 7, 6
objp= numpy.zeros((patw*path, 3))
for i in range(patw*path):
    objp[i, :2] = numpy.array([i % patw, i / patw], numpy.float32)
objp_list, imgp_list= [], []

while 1:
    stat, image = cap.read(0)
    ret, centers = cv2.findCirclesGrid(image, (patw, path), None)
    cv2.drawChessboardCorners(image, (patw, path), centers, ret)
    cv2.imshow('Camera', image)
    key = cv2.waitKey(10)
    if key == 0x1b:  # ESC
        break
    elif key == 0x20 and ret == True:
        print('Saved!')
        objp_list.append(objp.astype(numpy.float32))
        imgp_list.append(centers)

print(objp_list)
if len(objp_list) >= 3:
    K = numpy.zeros((3, 3), float)
    dist = numpy.zeros((5, 1), float)
    cv2.calibrateCamera(objp_list, imgp_list, (image.shape[1], image.shape[2]), K, dist)
    mtx= K.tolist()
    distor = dist.tolist()
    result = {
        "K": mtx,
        'Distortion': distor
    }
    print('K = ¥n', K)
    """numpy.savetxt('K.txt', K)"""
    print('Distcoeff= ¥n', dist)
    """numpy.savetxt('distCoef.txt', dist)"""
    with open('data/calibra.json', 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)

cap.release()
cv2.destroyAllWindows()
