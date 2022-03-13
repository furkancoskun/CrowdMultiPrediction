import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx 
from Models import CrowdCounting
from Models import CrowdMultiPrediction

dummy_input = torch.randn(1, 3, 1080, 1920, requires_grad=False)  

#resnet= models.resnet50(pretrained=True)  
resnet= models.resnet34(pretrained=False)  
modules = list(resnet.children())[:-2]
model = nn.Sequential(*modules)
model.eval() 
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "Resnet34.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
    ) 
print('resnet backbone has been converted to ONNX') 

# resnet.eval() 
# torch.onnx.export(resnet,         # model being run 
#         dummy_input,       # model input (or a tuple for multiple inputs) 
#         "resnet.onnx",       # where to save the model  
#         export_params=True,  # store the trained parameter weights inside the model file 
#     ) 

# crowdCountingModel= CrowdCounting()  
# crowdCountingModel.eval() 
# torch.onnx.export(crowdCountingModel,         # model being run 
#         dummy_input,       # model input (or a tuple for multiple inputs) 
#         "crowdCountingModel.onnx",       # where to save the model  
#         export_params=True,  # store the trained parameter weights inside the model file 
#     ) 
# print('crowdCountingModel has been converted to ONNX') 

