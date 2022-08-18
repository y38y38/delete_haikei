import shutil
#from google.colab import files
import os
import numpy as np
from PIL import Image


pretrained_ckpt = 'content/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'


input_folder = 'demo/image_matting/colab/input'
#if os.path.exists(input_folder):
#  shutil.rmtree(input_folder)
#os.makedirs(input_folder)

output_folder = 'demo/image_matting/colab/output'
#if os.path.exists(output_folder):
#  shutil.rmtree(output_folder)
#os.makedirs(output_folder)

background_folder = 'demo/image_matting/colab/background'
#if os.path.exists(background_folder):
#  shutil.rmtree(background_folder)
#os.makedirs(background_folder)

def combined_display(image, matte):
  # calculate display resolution
  w, h = image.width, image.height
  rw, rh = 800, int(h * 800 / (3 * w))
  
  # obtain predicted foreground
  image = np.asarray(image)
  if len(image.shape) == 2:
    image = image[:, :, None]
  if image.shape[2] == 1:
    image = np.repeat(image, 3, axis=2)
  elif image.shape[2] == 4:
    image = image[:, :, 0:3]
  matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
  foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
  haikeinasi = Image.fromarray(np.uint8(foreground))
  print(type(haikeinasi))
  # combine image, foreground, and alpha into one line
  combined = np.concatenate((image, foreground, matte * 255), axis=1)
  combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
  print(type(combined))
  return haikeinasi

image_names = os.listdir(input_folder)
print(image_names)
print(input_folder)
for image_name in image_names:
  matte_name = image_name.split('.')[0] + '.png'
  image = Image.open(os.path.join(input_folder, image_name))
  matte = Image.open(os.path.join(output_folder, matte_name))
#  display(combined_display(image, matte))
  image = combined_display(image, matte)
  print(image_name, '\n')
  image.save(image_name.split('.')[0] + '_non_back.png')
