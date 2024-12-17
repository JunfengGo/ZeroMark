'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torchvision.utils as tvls
import torch
import matplotlib.pyplot as plt 
import numpy as np

def save_tensor_images(images, filename, nrow = None, normalize = True):

    if not nrow:

        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)


if __name__ == "__main__" :
    
    torch.manual_seed(888)
    
    noise_1 = torch.rand(3,4,4)
    
    torch.manual_seed(444)
    
    noise_2 = torch.rand(3,4,4)

    torch.manual_seed(111)
    
    noise_3 = torch.rand(3,4,4)
    
    torch.manual_seed(222)
    
    noise_4 = torch.rand(3,4,4)

    torch.manual_seed(333)
    
    noise_5 = torch.rand(3,4,4)

    torch.manual_seed(555)
    
    noise_6 = torch.rand(3,4,4)

    torch.manual_seed(666)
    
    noise_7 = torch.rand(3,4,4)
    
    torch.manual_seed(1222)
    
    noise_4 = torch.rand(3,4,4)

    torch.manual_seed(1333)
    
    noise_5 = torch.rand(3,4,4)

    torch.manual_seed(1555)
    
    noise_6 = torch.rand(3,4,4)

    torch.manual_seed(1666)
    
    noise_7 = torch.rand(3,4,4)

    sample = torch.load("data_tensor/cifar_3.pth")

    for h in [50]:
        
        tensor = torch.load(f"./saved_per_vgg/per_3_0.pth")



        cosi = torch.nn.CosineSimilarity(dim=0) 
    
        delta_1 = torch.empty(200)
        delta_2 = torch.empty(200)
        delta_3 = torch.empty(200)
        delta_4 = torch.empty(200)
        delta_5 = torch.empty(200)
        delta_6 = torch.empty(200)
        delta_7 = torch.empty(200)
        delta_8 = torch.empty(200)
        delta_9 = torch.empty(200)
        delta_10 = torch.empty(200)
        delta_11 = torch.empty(200)
# -sample[i][: ,28:,28:]
        for i in range(200):
        
            tensor_flatten = torch.flatten(tensor[i][:,28:,28:])
        
            delta_flatten = torch.flatten(noise_1 - sample[i][:,28:,28:])
            delta_flatten2 = torch.flatten(noise_2 - sample[i][:,28:,28:])
            delta_flatten3 = torch.flatten(noise_3 - sample[i][:,28:,28:])
            delta_flatten4 = torch.flatten(noise_4 - sample[i][:,28:,28:])
            delta_flatten5 = torch.flatten(noise_5 - sample[i][:,28:,28:])
            delta_flatten6 = torch.flatten(noise_6 - sample[i][:,28:,28:])
            delta_flatten7 = torch.flatten(noise_7 - sample[i][:,28:,28:])
        
            score,active_index = torch.sort(torch.abs(tensor_flatten))
        # print(score)
        # print(active_index)
            active_index = active_index[-10:]
        # print(active_index)
            tensor_flatten = tensor_flatten[active_index]
        # print(tensor_flatten)
            delta_flatten = delta_flatten[active_index]
            delta_flatten2 = delta_flatten2[active_index]
            delta_flatten3 = delta_flatten3[active_index]
            delta_flatten4 = delta_flatten4[active_index]
            delta_flatten5 = delta_flatten5[active_index]
            delta_flatten6 = delta_flatten6[active_index]
            delta_flatten7 = delta_flatten7[active_index]


        # print(delta_flatten)
            output_1 = cosi(tensor_flatten.cuda(), delta_flatten.cuda())
        
  

            output_2 = cosi(tensor_flatten.cuda(), delta_flatten2.cuda())
            output_3 = cosi(tensor_flatten.cuda(), delta_flatten3.cuda())
            output_4 = cosi(tensor_flatten.cuda(), delta_flatten4.cuda())
            output_5 = cosi(tensor_flatten.cuda(), delta_flatten5.cuda())
            output_6 = cosi(tensor_flatten.cuda(), delta_flatten6.cuda())
            output_7 = cosi(tensor_flatten.cuda(), delta_flatten7.cuda())


        # print(output)
            delta_1[i] = output_1
            delta_2[i] = output_2
            delta_3[i] = output_3
            delta_4[i] = output_4
            delta_5[i] = output_5
            delta_6[i] = output_6
            delta_7[i] = output_7


    # tens = tensor[3].detach().cpu().numpy()
    
    # tens = tens.transpose(1,2,0)
    # trigger = torch.zeros((3,32,32))
    # trigger[:,28:,28:] = noise
    # trigger = trigger.detach().cpu().numpy()
    # trigger = trigger.transpose(1,2,0)

    # plt.imshow(1-trigger)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("watermark.pdf",bbox_inches='tight')
    # save_tensor_images   (tensor[2], "a.png", nrow = 1, normalize = True)
    # print(torch.sort(delta_1)[0][-8:-5])
        index = 100
        value = torch.sort(torch.sort(delta_1)[0][-index:]-
    1/6*(torch.sort(delta_2)[0][-index:]+
    torch.sort(delta_3)[0][-index:]+
    torch.sort(delta_4)[0][-index:]+
    torch.sort(delta_5)[0][-index:]+
    torch.sort(delta_6)[0][-index:]+
    torch.sort(delta_7)[0][-index:]))[0]
    #     index = torch.sort(delta_1)[1][-30:]
    #     print(delta_1[index])
    #     print(delta_2[index])
    #     value = delta_1[index]-1/6*(delta_2[index]+
    # delta_3[index]+
    # delta_4[index]+
    # delta_5[index]+
    # delta_6[index]+
    # delta_7[index])
        print(value)

      

        torch.save(value,"tensor_plot_b/value_vgg_3_0.pth")
