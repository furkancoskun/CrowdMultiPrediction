import numpy as np
import cv2
import numpy as np
import json
from DensityMapGaussian import getGaussianDensityMap

frame = cv2.imread("/home/frkncskn/Desktop/TestVideos/GTA_EVENTS/0/0/0000000000.tiff")
frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_AREA).astype(np.float32)

json_file_path = "/home/frkncskn/Desktop/TestVideos/GTA_EVENTS/0/0/0000000000.json"
coord_list = []
count = 0
with open(json_file_path) as f:
    json_file = json.load(f)
    for j in json_file['Detections']:
        if "IK_Head" not in j["Bones"]:
            continue
        count = count +1
        x_coord = round(j["Bones"]["IK_Head"]["X"]*2560)
        y_coord = round(j["Bones"]["IK_Head"]["Y"]*1440)
        coord_list.append({'x':x_coord, 'y':y_coord})

densityMap = getGaussianDensityMap(frame, coord_list, 0.75, 25, 4.0)
densityMap = densityMap * 255
print(densityMap.max())
print(densityMap.shape)
print(densityMap.dtype)
cv2.imshow("densityMap", densityMap) 
k = cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

    