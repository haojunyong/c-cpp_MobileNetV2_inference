# MobileNetV2_inference_C++
MobileNetV2 inference C++ basic implement
从零用C++实现MobileNetV2

# How to start inference

test.cpp main()
其中`mobilenet_inference1()`为单张图片推断
```C++
char image[100] = "./test_image/n0761348000000310.jpg";
mobilenet_inference1(&net, image);
```
`mobilenet_inference()`为多张图片推断

```C++
mobilenet_inference(&net, 10);//填测试图片数量（小于等于2000张)
```

# pretrained model
torchvision
```python
from torchvision import models
net = models.mobilenet_v2(pretrained=True)
```
save as `MobileNetV2_pretrained.data`

# images pro-processing

from mini-imagenet test set
randomly choose 2000 images
and did pro-processing as shown below

```python
from PIL import Image 
from torchvision import transforms
def img_to_224(before,after):
	input_size=224
    trans=transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
    ])
    img=Image.open(before)
    img=trans(img)
    img.save(after)
```

# operator implement
## conv
just 6 for loops which is simple but not good
about to change it to im2col

## batch inference
plan to implements

