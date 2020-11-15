# JESRL-FER
The code for the paper "Joint Expression Synthesis and Representation Learning for Facial Expression Recognition"

## Pre-requisites
 (1) Python 3.6.7.
 
 (2) Scipy.
 
 (3) PyTorch (r1.0.1) .
 

##  Datasets
 (1) You may use any dataset with labels of the expression. 
 In our experiments, we use <br/>
 Multi-PIE (http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html) <br/>
 RAF-DB (http://www.whdeng.cn/RAF/model1.html). <br/>
 
 (2) It is better to detect the face before you train the model. In this paper, we use a lib face detection algorithm (https://github.com/ShiqiYu/libfacedetection)

Besides, please ensure that you have the following directory tree structure in your repo.<br/>
├── datasets<br/>
│   └── multiple<br/>
│       ├──── data<br/>
│       ├──── images_test.list<br/>


## Traininig

To train the facial expression synthesis and facial expression recognition model on the Multi-PIE dataset, you can run the following code.
Here `-a disgust scream smile squint surprise neutral` indicates the expression names.
Notes. Please do not change the order of expressions in `-a disgust scream smile squint surprise neutral`

```
$ python train.py -a disgust scream smile squint surprise neutral --mode train --multi_add_gan --alpha5 --gpu 0 
```

## Testing
#### 1.Facial expression synthesis

To swap the expressions between two unpaired images, you can run the following code. And `--swap_list 3 6` means the expression id of the input image and target image, respectively. The generated image is saved as `result.jpg`, which includes the original images and generated images with exchanged expressions.

```
$ python train.py -a disgust scream smile squint surprise neutral --swap_list 2 5 --input ./images/smile.jpg --target ./images/neutral.jpg --mode test --restore --gpu 0
```

#### 2.Facial Expression Recognition

To evaluate the facial expression recognition model, you can run the following  code.
```
$ python test_matrix_multi.py --data 2 --resume_dir checkpoint/checkpoint_multipie/VGG --gpu 0 # run on the Multi-PIE dataset
```
