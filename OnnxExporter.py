import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx 

resnet50= models.resnet50(pretrained=False)  
#resnet50= models.resnet34(pretrained=False)  
model = nn.Sequential(
    resnet50.conv1,
    resnet50.bn1,
    resnet50.relu,
    resnet50.maxpool,
    resnet50.layer1,
    resnet50.layer2,
    resnet50.layer3,
    resnet50.layer4
)

model.eval() 
dummy_input = torch.randn(1, 3, 1440, 2560, requires_grad=False)  

torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "Resnet50.onnx",       # where to save the model  
        export_params=False,  # store the trained parameter weights inside the model file 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
    ) 

print(" ") 
print('Model has been converted to ONNX') 