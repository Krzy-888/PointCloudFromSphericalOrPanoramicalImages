from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
# import numpy as np
# import os
gps_logs = []
images_name = []
for i in range(56 - 19):
    images_name.append(f"V1{i+54}02{i+20}.JPG")
for i in range(len(images_name)):
    img = Image.open(images_name[i])
    exif = img._getexif()
    gps_data = {}
    for tag, value in exif.items():
        tag_name = TAGS.get(tag)
        if tag_name == "GPSInfo":
            for key in value:
                gps_data[GPSTAGS.get(key)] = value[key]
                
            # print(gps_data['GPSLatitude'],gps_data['GPSLongitude'])
            value_lat = (gps_data['GPSLatitude'][0]+gps_data['GPSLatitude'][1]/60 + gps_data['GPSLatitude'][2]/3600)
            value_lon = (gps_data['GPSLongitude'][0]+gps_data['GPSLongitude'][1]/60 + gps_data['GPSLongitude'][2]/3600)
            print(images_name[i],'\t',float(value_lat),'\t',float(value_lon))
            gps_logs.append(f'{images_name[i]},{float(value_lat)},{float(value_lon)}\n')
with open('data.csv','w') as log:
    # for row in gps_logs:
    # log.writelines(gps_logs)
    log.writelines(gps_logs)
print("done")