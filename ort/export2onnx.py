import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

if __name__ == "__main__":
    # import the arguments
    import argparse
    parser = argparse.ArgumentParser("Object Segmentation on ONNX Runtime")
    parser.add_argument('--name', type=str, help='Name of the ONNX model.')
    parser.add_argument('--height', type=int, default=224, help='Height of the camera input.')
    parser.add_argument('--width', type=int, default=224, help='Width of the camera input.')
    args = parser.parse_args()

    # load model
    class ObjectSegmentation(nn.Module):
        def __init__(self, pretrained_model, output_key="out"):
            super(ObjectSegmentation, self).__init__()
            self.pretrained_model = pretrained_model
            self.output_key = output_key

        def forward(self, x):
            output = self.pretrained_model(x)[self.output_key]
            softmax_output = F.softmax(output, dim=1)

            return softmax_output
    
    obj_seg = models.segmentation.lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    model = ObjectSegmentation(obj_seg, 'out')
    model = model.eval()

    # create a random input
    input = torch.randn(1, 3, args.height, args.width)

    # analyze the outputs
    with torch.no_grad():
        outputs = model(input)

    # convert to the onnx model
    torch.onnx.export(
        model, # YOLOP model
        input, # YOLOP input
        args.name, # Directory of the onnx model
        export_params=True, # store the model
        do_constant_folding=True, # Optimize the model
        opset_version=17, # ONNX version
        input_names=['input'], # the model's input names
        output_names=['output'] # the model's output names
    )