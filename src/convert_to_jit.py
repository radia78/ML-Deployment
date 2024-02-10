import torch
from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
from PIL import Image

def load_model():
    # load the model
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    return model.eval()

def convert_to_jit(model_name:str, sample_img, model) -> None:
    # convert the PyTorch model to JIT model for C++ inference
    jit_model = torch.jit.trace(model, sample_img, strict=False)
    jit_model.save(model_name)

def img_to_tensor(img_path:str):
    # convert the image into tensors
    img = Image.open(img_path).convert("RGB")
    img_transforms = LRASPP_MobileNet_V3_Large_Weights.DEFAULT.transforms()
    print(img_transforms)
    return img_transforms(img).unsqueeze(0)

def main(img_path:str, pt_model:str) -> None:
    model = load_model()
    sample_img = img_to_tensor(img_path)
    convert_to_jit(pt_model, sample_img, model)

if __name__ == "__main__":
    img_path = "/Users/radiakbar/Projects/torch_cpp/asset/sample_img.png"
    pt_model = "/Users/radiakbar/Projects/torch_cpp/asset/seg_model.pt"
    main(img_path=img_path, pt_model=pt_model)
    print(LRASPP_MobileNet_V3_Large_Weights.DEFAULT.meta['categories'])