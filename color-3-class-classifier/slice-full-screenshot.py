from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import math

image_size = 32
pixel_depth = 255.0

screen_width = 800
screen_height = 600

x_tiles = math.trunc(screen_width / image_size) -1;
y_tiles = math.trunc(screen_height / image_size) -1;

print("x_tiles = %d" % x_tiles)
print("y_tiles = %d" % y_tiles)
data_root = '.'

image_path = './StarCraft_200325_as.png'

def load_screen(image_path):
    image_data = (ndimage.imread(image_path, False, 'RGB').astype(float) -
              pixel_depth / 2) / pixel_depth
    return image_data

image_data = load_screen(image_path)

dataset = np.ndarray(shape=(x_tiles * y_tiles, image_size, image_size, 3),
                             dtype=np.float32)
for j in range (0, y_tiles):
    for i in range(0, x_tiles):
        dataset[j * x_tiles + i] = image_data[j*image_size:j*image_size + image_size, i * image_size : i *image_size + image_size, :]

def write_png(buf, width, height):
    """ buf: must be bytes or a bytearray in Python3.x,
        a regular string in Python2.x.
    """
    import zlib, struct

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

data = write_png(dataset[0], 32, 32)
with open("my_image.png", 'wb') as fd:
    fd.write(data)

print(image_data.shape)

def render_image_ascii(image_data):
    for j in range (0, y_tiles):
        line = ''
        for i in range (0,x_tiles):
            teal_present = False
            for k in range (i * image_size, i*image_size + image_size):
                for l in range (j* image_size, j*image_size + image_size):
                    if image_data[l][k][1] > 0.15:
                        teal_present = True
            if (teal_present):
                line = line + '#'
            else:
                line = line + '.'
        print(line)

render_image_ascii(image_data)

pickle_file = os.path.join(data_root, 'slicedScreen.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'test_dataset': dataset,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
