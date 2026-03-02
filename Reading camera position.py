from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
gps_logs = []
images_name = []

for i in range(len(images_name)):
    img = Image.open(images_name[i])
    exif = img._getexif()
    gps_data = {}
    for tag, value in exif.items():
        tag_name = TAGS.get(tag)
        if tag_name == "GPSInfo":
            for key in value:
                gps_data[GPSTAGS.get(key)] = value[key]
                gps_logs.append(gps_data)

