from skimage import data, io
import numpy as np
import os
import sys

data_size = 415
pic = []
test_image = ['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg'] 
image_path = sys.argv[1]        # Aberdeen/
pic_num = sys.argv[2]           # 87.jpg
ans_name = sys.argv[3]          # 87_reconstruct
# Number of principal components used
k = 5

def process(M): 
    M = np.copy(M)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

for i in range(data_size):
    name = image_path + str(i) + ".jpg"
    img = io.imread(name)
    pic.append(img)

pic = np.array(pic).astype('float32')
pic = np.reshape(pic, (data_size, 600*600*3))

mean = np.sum(pic, axis=0) / data_size
pic -= mean                                 # 415, 600*600*3


u, s, v = np.linalg.svd(pic.transpose(), full_matrices = False)         #600*600*3, 415
print("mean :", mean.shape)
print("pic :", pic.shape)
print("u :", u.shape)                   # 1080000 * 415
print("s :", s.shape)
print("v :", v.shape)


# draw mean face
average = process(mean)
io.imsave('average.jpg', average.reshape((600, 600, 3)) )  


# draw eigenfaces
for i in range(k):
    eigenface = process(u.transpose()[i])
    io.imsave(str(i) + '_eigenface.jpg', eigenface.reshape((600, 600, 3)) )


# reconstruct----------------------------------------------------------------------------

# Load image & Normalize
picked_img = io.imread(os.path.join(image_path,pic_num))  
X = picked_img.flatten().astype('float32') 
X -= mean

# Compression
weight = np.array([u.transpose()[i].dot(X) for i in range(k)])         
# Reconstruction
reconstruct = process( weight.dot(u.transpose()[:k]) + mean )

io.imsave(ans_name, reconstruct.reshape((600, 600, 3)))
#-----------------------------------------------------------------------------------------

# print five biggest eigenvalue ratio
for i in range(k):
    number = s[i] * 100 / np.sum(s)
    print(number)



