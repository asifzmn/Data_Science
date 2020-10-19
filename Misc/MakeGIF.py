import os
from PIL import Image

if __name__ == '__main__':
    fp_in = "/home/az/Desktop/2018 Weekly"
    fp_out = "/home/az/Desktop/AirQuality.gif"
    left, top, right, bottom = 75, 150, 850, 1075
    images = [Image.open(os.path.join(fp_in,n)).crop((left, top, right, bottom)) for n in sorted(os.listdir(fp_in))]
    images[0].save(fp_out,save_all=True,append_images=images[1:],duration=750,loop=0)