import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx 

resnet18 = models.resnet18(pretrained = True)
model = nn.Sequential(
    resnet18.conv1,
    resnet18.bn1,
    resnet18.relu,
    resnet18.layer1
)

#model = models.resnet18(pretrained = True)

model.eval() 
dummy_input = torch.randn(1, 3, 320, 180, requires_grad=True)  

torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "ResnetCrowdBackbone.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 

print(" ") 
print('Model has been converted to ONNX') 