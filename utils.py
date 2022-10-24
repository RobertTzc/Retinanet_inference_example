from pyexiv2 import Image
def read_LatLotAlt(image_dir):
    info = Image(image_dir)
    exif_info = info.read_exif()
    xmp_info = info.read_xmp()
    re = dict()
    re['latitude'] = float(xmp_info['Xmp.drone-dji.GpsLatitude'])
    re['longitude'] = float(xmp_info['Xmp.drone-dji.GpsLongitude'])
    re['altitude'] = float(xmp_info['Xmp.drone-dji.RelativeAltitude'][1:])
    #print (image_name,xmp_info['Xmp.drone-dji.RelativeAltitude'])
    return re
info = read_LatLotAlt('/home/zt253/Models/Retinanet_inference_example/example_images/Bird_B/DJI_0328.JPG')
print (info)