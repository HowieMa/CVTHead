## A Summary of all items within this folder: 

~~~
# The following items are already included: 
mean_deform.pt      # for Hair/shoulder deformation model 
u_full.pt           # Hair/shoulder deformation model 
transform.pkl       # downsample/upsample mesh from 5024 vertices to 314 vertices, for Spiralnet
head_temmplate.obj  # FLAME face model

# Yo need to download the following items
cvthead.pth                 # our pretrained CVT Head on VoxCeleb1
79999_iter.pth              # Face Parsing model from https://github.com/VisionSystemsInc/face-parsing.PyTorch
linear_hair.pth             # Hair/shoulder deformation model from ROME
rome.pth                    # pre-trained ROME, we will use its CNN feature extractor
generic_model.pkl               # FLAME model, obtained from https://flame.is.tue.mpg.de/
resnet50_scratch_weight.pth     # pretrained ResNet-50 on VGGFace2, for face identity loss
deca_model.tar                  # pre-trained DECA
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

- ROME
[rome.pth](https://drive.google.com/file/d/1rLtc037Ra6Z6t0kp-gJ8P1ZKfzkKm070/view)

You can also find all these files at:
[rome](https://github.com/SamsungLabs/rome)


#### DECA
- [DECA](https://github.com/yfeng95/DECA/tree/master)

#### FLAME
- [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)

#### Face Identity Loss with pre-trained ResNet50 on VGGFace2
- [resnet50_scratch_weight.pth](https://drive.google.com/file/d/17bGCDQLuXU81xqHF1MB6nBqpBO6PtPd2/view?usp=sharing)
You can also find it at: [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch/tree/master)