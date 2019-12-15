# offical pytorch implement of seesawfacenet

------

## 1. Intro

- This repo is a reimplementation of seesawfacenet[(paper)](https://arxiv.org/abs/1908.09124)
- For models, including the pytorch implementation of the backbone modules of Arcface/MobileFacenet/seesawfacenet(including seesaw_shareFaceNet,seesaw_shuffleFaceNet, DW_seesawFaceNetv1, DW_seesawFaceNetv2)
- We build build this repo based on the work of @TreB1eN(https://github.com/TreB1eN/InsightFace_Pytorch), and fixed few bugs before usage.
- Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) in the original paper

------

## 2. Pretrained Models & training logs & Performance

[seesawfacenet @ googledrive](https://drive.google.com/drive/folders/1n4Zi7YTqG4YoLdK3-aO8qWWEjOCcD7w9?usp=sharing)
![Image text](https://github.com/cvtower/seesawfacenet_pytorch/raw/master/figures/mobile_version.jpg)
![Image text](https://github.com/cvtower/seesawfacenet_pytorch/raw/master/figures/dw_version.jpg)

## 3. How to use

- clone

  ```
  git clone https://github.com/TropComplique/mtcnn-pytorch.git
  ```

### 3.1 Data Preparation

#### 3.1.1 Prepare Facebank (For testing over camera or video)

Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:

```
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```

#### 3.1.2 download the pretrained model to work_space/model

If more than 1 image appears in one folder, an average embedding will be calculated

#### 3.2.3 Prepare Dataset (MS1MV2(face_emore, refined MS1M...whatever we call it) For training refer to the original paper)

download the MS1MV2 dataset:

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ), [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- More Dataset please refer to the [original post](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

**Note:** If you use [MS1MV2](https://arxiv.org/abs/1607.08221) dataset and the cropped [VGG2](https://arxiv.org/abs/1710.08092) dataset, please cite the original papers.

- after unzip the files to 'data' path, run :

  ```
  python prepare_data.py
  ```

  after the execution, you should find following structure:

```
faces_emore/
            ---> agedb_30
            ---> calfw
            ---> cfp_ff
            --->  cfp_fp
            ---> cfp_fp
            ---> cplfw
            --->imgs
            ---> lfw
            ---> vgg2_fp
```

------

### 3.2 detect over camera:

- 1. download the desired weights to model folder:

- 2 to take a picture, run

  ```
  python take_pic.py -n name
  ```

  press q to take a picture, it will only capture 1 highest possibility face if more than 1 person appear in the camera

- 3 or you can put any preexisting photo into the facebank directory, the file structure is as following:

```
- facebank/
         name1/
             photo1.jpg
             photo2.jpg
             ...
         name2/
             photo1.jpg
             photo2.jpg
             ...
         .....
    if more than 1 image appears in the directory, average embedding will be calculated
```

- 4 to start

  ```
  python face_verify.py 
  ```

- - -

### 3.3 detect over video:

```
​```
python infer_on_video.py -f [video file name] -s [save file name]
​```
```

the video file should be inside the data/face_bank folder

previous work on mtcnn for android platform and face cropping
- mtcnn_android_native [mtcnn_android_native](https://github.com/cvtower/mtcnn_android_native)
- Face-extractor-based-on-mtcnn [Face-extractor-based-on-mtcnn](https://github.com/cvtower/Face-extractor-based-on-mtcnn)

### 3.4 Training:

```
​```
python train.py -b [batch_size] -lr [learning rate] -e [epochs]

# python train.py -net mobilefacenet -b 256 -w 24
​```
```

## 4. References

- This repo is mainly based on [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) and [cvtower/SeesawNet_pytorch](https://github.com/cvtower/SeesawNet_pytorch), and inspired by [deepinsight/insightface](https://github.com/deepinsight/insightface) as well.

## PS

- PRs are welcome, especially for models for mobile platfroms
- Email : jtzhangcas@gmail.com
