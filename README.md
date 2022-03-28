# yolov1_maxim
This repo covers the MAX78000 model training and synthesis pipeline for the YOLO v1 model.

This is a slightly modified version of the original repo by Guanchu Wang (gw22@rice.edu) & Yi-Wei Chen (yiwei_chen@tamu.edu).

Changes:
- moved some files into the history folder that were not in use
- changed some static paths for the datasets
- added argument parser to the YOLO_V1_Test.py python script

---

### Role of each Python script:

* YOLO_V1_Train_QAT.py: Python script used to train the network

* YOLO_V1_Test.py: Fake INT8 test of the model; change the directory of weight file (*.pth) to test different models.

* YOLO_V1_DataSet_small.py: Preprocess the VOC2007 dataset.

* yolov1_bn_model.py: Define the structure of the deep neural network.

* YOLO_V1_LossFunction.py: Define the loss function.


---

### Environment setup
Note that Python >= 3.8
```bash
$ git clone git@github.com:YIWEI-CHEN/yolov1_maxim.git
$ cd yolov1_maxim
# note that ai8x-training and ai8x-synthesis should in the project root (e.g., yolov1_maxim)
$ git clone --recursive https://github.com/MaximIntegratedAI/ai8x-training.git
$ git clone --recursive https://github.com/MaximIntegratedAI/ai8x-synthesis.git

# in your virtual environment
# install pytorch for NVIDIA RTX A5000
$ pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# install distiller
$ cd yolov1_maxim/ai8x-training/distiller
# remove lines of numpy, torch, torchvision in requirements.txt
$ pip install -e .

# install other packages
$ pip install tensorboard matplotlib numpy colorama yamllint onnx PyGithub GitPython opencv-python
```


---

### How to train the YOLO model?

After initializing the repo as described above

Run 'python YOLO_V1_Train_QAT.py'.

    * You can change the hyperparameter as you want. But there is no need to do this because the current hyperparameters work for our Layer-wise QAT training.


---

### How to test the trained model (Fake INT8 testing)?

1. Open YOLO_V1_Test.py, revise line 27 into the directory of your trained model.

2. Run YOLO_V1_Test.py. (python3 YOLO_V1_Test.py or using Pycharm)

---

### How to generate the checkpoint file of our model?

Run python YOLO_V1_Test.py --gen_checkpoint to generate the distilled checkpoint file. Afterwards you can quantize the checkpoint using ai8x-synthesis.


---

### Trained models

The follow links contains previous trained models and logs.

1. [weight_20210711](https://drive.google.com/drive/folders/1vq-7v-ALpb-Rja-A26G-X3aW25lZYPr6?usp=sharing)

2. [yolo_models_test](https://drive.google.com/drive/folders/1i2Wiom7VP05wWcpyN4yMaJavjDQnS49T?usp=sharing) 

3. [logs](https://drive.google.com/drive/folders/1gHSb_aIbfadDJwKjGwqWi9tjVzLNtzx0?usp=sharing)


---
### VOC 2007 Dataset
* You can download [train/validation](http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar) 
  and [test](http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar) by the above hyperlinks.
* Or if you have an Texas A&M account, you might access the VOC2007 on datalab6.engr.tamu.edu
   * Train: /data/yiwei/VOC2007/Train
   * Test: /data/yiwei/VOC2007/Test

* Place the downloaded datasets in the dataset/Train and dataset/Test folder

---
### Reference
* The repo starts from https://www.dropbox.com/s/pssda2gxrqa51v9/yolov1_maxim.zip?dl=0
* The YOLOv1 train and test framework are from https://github.com/ProgrammerZhujinming/YOLOv1

---
### Contributors

* Guanchu Wang (gw22@rice.edu)
  
* Yi-Wei Chen (yiwei_chen@tamu.edu)

* Jonas Menges (dev.menges.jonas@gmail.com)
