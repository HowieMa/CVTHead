## A Summary of all items within this folder: 

~~~
# CVT Head
cvthead.pth 

# Face Parsing
79999_iter.pth 

# Hair/shoulder deformation model
linear_hair.pth 
mean_deform.pt  (included)
u_full.pt  (included)   

# ROME (CNN feature extractor)
rome.pth

# FLAME 
transform.pkl   # downsample/upsample mesh from 5024 vertices to 314 vertices (included)
generic_model.pkl

# DECA
deca_model.tar

# face identity loss (pretrained ResNet-50 on VGGFace2 datasets)
resnet50_scratch_weight.pth 
~~~


## Downloads

You can download all files by: 

~~~
bash fetch_data.sh
~~~

This scripts will automatically download all required files


In case it doesn't work, you can also download each file separately from their project pages:
 

#### face parsing
- [79999_iter.pth](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)

You can also find this file at:
[face-parsing.PyTorch](https://github.com/VisionSystemsInc/face-parsing.PyTorch)

#### ROME
- Linear deformation model
[linear_hair.pth](https://drive.google.com/file/d/1Enw9MU9Xin77ws08y4pNqkMW0AyUIzv_/view)

- ROME CNN feature extractor
[rome.pth](https://drive.google.com/file/d/1rLtc037Ra6Z6t0kp-gJ8P1ZKfzkKm070/view)

You can also find all these files at:
[rome](https://github.com/SamsungLabs/rome)


#### DECA
- [DECA](https://github.com/yfeng95/DECA/tree/master)

#### FLAME
- [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)

#### Face Identity Loss with pre-trained ResNet50 on VGGFace2
[resnet50_scratch_weight.pth](https://drive.google.com/file/d/17bGCDQLuXU81xqHF1MB6nBqpBO6PtPd2/view?usp=sharing)
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch/tree/master)