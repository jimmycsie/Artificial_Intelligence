import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import sys

resnet50 = models.resnet50(pretrained=True) #download and load pretrained model
resnet50.eval()

def visualize(x_adv, output_path):
    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))
    x_adv = np.clip(x_adv, 0, 1)

    x_adv = np.round(x_adv*255)
    pic = Image.fromarray(x_adv.astype("uint8"))
    pic.save(output_path)


# const variable ----------------------------------
pic_num = 200
class_num = 1000
# create name format ----------------------------------
number = []
for i in range(10):
    number.append("00"+str(i))
for i in range(10, 100):
    number.append("0"+str(i))
for i in range(100, 200):
    number.append(str(i))
#------------------------------------------------------
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
epsilon = 0.011
num_steps = 5
alpha = 0.024
preprocess = transforms.Compose([
                transforms.Resize((224,224)),  
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])



for i in range(pic_num):
    img_path = sys.argv[1] + '/' + number[i] + ".png"
    pic = Image.open(img_path)
    pic =  pic.convert('RGB')           # change to jpg format

    image_tensor = preprocess(pic) #preprocess an i
    image_tensor = image_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W
    img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable

    # get label
    output = resnet50.forward(img_variable)
    label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
    if(i%10==0):
        print(i)

    y_true = Variable( torch.LongTensor([label_idx]), requires_grad=False)   #tiger cat
    for j in range(num_steps):
        zero_gradients(img_variable)                       #flush gradients
        output = resnet50.forward(img_variable)         #perform forward pass
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, y_true)
        loss_cal.backward()
        x_grad = alpha * torch.sign(img_variable.grad.data)   # as per the formula
        adv_temp = img_variable.data + x_grad                 #add perturbation to img_variable which also contains perturbation from previous iterations
        total_grad = adv_temp - image_tensor                  #total perturbation
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)
        x_adv = image_tensor + total_grad                      #add total perturbation to the original image
        img_variable.data = x_adv


    output_adv = resnet50.forward(img_variable)
    x_adv_pred = torch.max(output_adv.data, 1)[1][0]  #classify adversarial example
    output_adv_probs = F.softmax(output_adv, dim=1)

    output_path= sys.argv[2] + '/' + number[i] + ".png"
    visualize(img_variable.data,output_path)




