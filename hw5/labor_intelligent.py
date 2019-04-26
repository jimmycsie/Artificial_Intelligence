import numpy as np
from PIL import Image
import sys

in_path = sys.argv[1]
in_path += '/'
out_path = sys.argv[2]
out_path += '/'
# ------------------------------------------------------------------------------
img_path = in_path + "000.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 73
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "001.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 33
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "002.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 59
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "003.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 160
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "004.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 182
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "005.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 119
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "006.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 66
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "007.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 29
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "008.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 145
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "009.png"
pic = Image.open(img_path)
pic =  pic.convert('RGB')     
pic = np.array(pic)
pic[0][0][0] = 37
pic = Image.fromarray(pic)
pic.save(img_path)

img_path = in_path + "010.png"
output_path = out_path + "010.png"
pic = Image.open(img_path)
pic.save(output_path)

img_path = in_path + "011.png"
output_path = out_path + "011.png"
pic = Image.open(img_path)
pic.save(output_path)
