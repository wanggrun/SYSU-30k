# SYSU-30k dataset, code, and pretrained model

The Dataset, code, and pretrained model of "Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark" https://arxiv.org/abs/1904.03845.

## Updated News
- [2022-05-14] According to our experiments in self-supervised representation learning, we found that SYSU-30k can examine the quality of SSL algorithms. The code for training SSL can be found at [here](https://github.com/wanggrun/triplet). The code to test SSL is close to the test code in this project. Currently, an SSL method has achieved a 27.5% rank-1 accuracy on the SYSU-30k benchmark!
- [2021-11-04] The SYSU-30k dataset is also uploaded to Google driver. Both Google driver and Baidu pan are availble now.


## The source code of our weakly supervised re-ID is originally written by [**Guangcong Wang**](https://wanggcong.github.io) who has rich experiences in person re-ID ([follow his github](https://github.com/Wanggcong)), and is partially revised by [Guangrun Wang](https://wanggrun.github.io/).

## Leaderboard on SYSU-30k. 

The leaderboard is also at ["paperswithcode.com"](https://paperswithcode.com/sota/person-re-identification-on-sysu-30k)
  
| Supervision      | Method       |  Rank-1 | 
|:------------------:|:------------------:|:------------------:|
| Generalization   | [DARI [1]](https://arxiv.org/abs/1604.04377)        |        11.2  |
| Generalization   |  [DF [2]](https://arxiv.org/abs/1512.03622)   |   10.3     | 
| Generalization   |  [ResNet-50 [3]](https://arxiv.org/abs/1512.03385)      |     20.1         |
| Generalization   | [Local-CNN [4]](https://dl.acm.org/doi/10.1145/3240508.3240645)       |     23.0         |
| Generalization   | [MGN [5]](https://arxiv.org/abs/1804.01438)             |     23.6         |
| Generalization   | [IICS [6]](https://arxiv.org/abs/2103.11658)         |     36.0         |
| Weakly Supervised  | [DGL [7]](https://arxiv.org/abs/1904.03845)      |     26.9         |
| Self Supervised   | [SimCLR [8]](https://arxiv.org/abs/2002.05709)          |     10.9         |
| Self Supervised   | [MoCo v2 [9]](https://arxiv.org/abs/2003.04297)        |     11.6         |
| Self Supervised   | [BYOL [10]](https://arxiv.org/abs/2006.07733)         |     12.7         |
| Self Supervised   | [Triplet [11]](https://arxiv.org/abs/2104.08760)         |     14.8         |

[1] Guangrun Wang, Liang Lin, Shengyong Ding, Ya Li, Qing Wang: DARI: Distance Metric and Representation Integration for Person Verification. AAAI 2016: 3611-3617

[2] Shengyong Ding, Liang Lin, Guangrun Wang, Hongyang Chao: Deep feature learning with relative distance comparison for person re-identification. Pattern Recognit. 48(10): 2993-3003 (2015)

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

[4] Jiwei Yang, Xu Shen, Xinmei Tian, Houqiang Li, Jianqiang Huang, Xian-Sheng Hua: Local Convolutional Neural Networks for Person Re-Identification. ACM Multimedia 2018: 1074-1082

[5] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou: Learning Discriminative Features with Multiple Granularities for Person Re-Identification. ACM Multimedia 2018: 274-282

[6] Shiyu Xuan, Shiliang Zhang: Intra-Inter Camera Similarity for Unsupervised Person Re-Identification. CVPR 2021: 11926-11935

[7] Guangrun Wang, Guangcong Wang, Xujie Zhang, Jianhuang Lai, Zhengtao Yu, Liang Lin: Weakly Supervised Person Re-ID: Differentiable Graphical Learning and a New Benchmark. IEEE Trans. Neural Networks Learn. Syst. 32(5): 2142-2156 (2021)

[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey E. Hinton: A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020: 1597-1607

[9] Xinlei Chen, Haoqi Fan, Ross B. Girshick, Kaiming He: Improved Baselines with Momentum Contrastive Learning. CoRR abs/2003.04297 (2020)

[10] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Ávila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning. NeurIPS 2020

[11] Guangrun Wang, Keze Wang, Guangcong Wang, Philip H.S. Torr, and Liang Lin, "Solving Inefficiency of Self-supervised Representation Learning" [C]. in Proc. of IEEE International Conference on Computer Vision (ICCV), 2021.

## Statistic of the dataset

 SYSU-30k contains 30k categories of persons, which is about 20 times large rthan CUHK03 (1.3k categories)and Market1501 (1.5k categories), and 30 times larger than ImageNet (1k categories). SYSU-30k contains 29,606,918 images. Moreover, SYSU-30k provides not only a large platform for the weakly supervised ReID problem but also a more challenging test set that is consistent with the realistic setting for standard evaluation. Figure 1 shows some samples from the SYSU-30k dataset. 
 
 
 
 ### Table 1: Comparision with existing Re-ID datasets.

| Dataset      | CUHK03       |  Market-1501 |   Duke       |      MSMT17  |       CUHK01 |         PRID |        VIPeR |       CAVIAR |      SYSU-30k|
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| Categories   | 1,467        | 1,501.       |   1,812      |      4,101   |        971    |        934  |        632   |        72    |      30,508  |
|   Scene      |    Indoor    |     Outdoor  |   Outdoor   |Indoor, Outdoor|    Indoor     |   Outdoor   |    Outdoor   |   Indoor     |Indoor, Outdoor|
|   Annotation |    Strong    |     Strong   |   Strong     |  Strong      |    Strong     |  Strong     |   Strong     |    Strong    |  Weak         |
| Cameras      |     2        |    6         |     8        |      15      |     10        |     2       |       2      |     2        |  Countless    |
|  Images      |     28,192   |   32,668     |    36,411    |   126,441    | 3,884         |    1,134    |      1,264   |     610      | 29,606,918    |


### Table 2:  Comparison with ImageNet-1k.
 
  
 | Dataset      | ImageNet-1k       |  SYSU-30k | 
|:------------------:|:------------------:|:------------------:|
| Categories   | 1,000        |        30,508  |
|  Images      |  1,280,000   |   29,606,918     | 
|  Annotation  |     Strong   |     Weak         |

### Figure 1: The statistics of SYSU-30k. 
 

<p align="center">
<img src = "https://github.com/wanggrun/SYSU-30k/blob/master/sysu-30k-statistics.png", width='300'>
 </p>

<p align='center'>(a) summarizes the number of the bags with respect to the number of the images per bag. (b) and (c) compare SYSU-30k with the existing datasets in terms of image number and person IDs for both the entire dataset and the test set.</p>


## Visualization of the dataset

### Figure 2: Example in SYSU-30k.

<p align="center">
<img src="https://github.com/wanggrun/SYSU-30k/blob/master/sysu-30k-example.png", width = '400'>
 </p>

 <p align='center'>(a) training images in terms of bag; (b) their bag-level annotations; (c) test set.</p>
 
 

## Download the dataset

**Note** that our original training set occupies 462G's memory. We are not able to upload the original data taking up such a large memory. As a result, we downsample the train images from 288 * x resolution to 144 * x resolution with x representing the shortest edge. The compressed data sum up to 82.2G.

The test set is uncompressed due to the appropriate memory size. We provide two ways to download the SYSU-30k dataset.  

### (1) Download the dataset on Google driver
Click [:arrow_down:](https://drive.google.com/drive/folders/1MTxZ4UN_mbxjByZgcAki-H10zDzzeyuJ?usp=sharing) for downloading.
The folder looks like:

```
sysu-30k-released
├── train
|   ├── sysu_train_set_all_part1.tar
|   ├── sysu_train_set_all_part2.tar
|   ├── sysu_train_set_all_part3.tar
|   ├── sysu_train_set_all_part4.tar
|   ├── sysu_train_set_all_part5.tar
|   ├── sysu_train_set_all_part6.tar
|   ├── sysu_train_set_all_part7.tar
|   ├── sysu_train_set_all_part8.tar
|   ├── sysu_train_set_all_part9.tar
|   ├── sysu_train_set_all_part10.tar

├── test
|   ├── sysu_test_set_all.tar
├── train.txt (for weakly supervised training, "filename\n" in each line)
├── query.txt (for evaluation)
├── gallery.txt (for evaluation)
├── net_6.pth (pretrained model)
```


### (2) Or download the dataset on Baidu driver

#### Download the training set

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  sysu_train_set_all_part1.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part2.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part3.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part4.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part5.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part6.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part7.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part8.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part9.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|  sysu_train_set_all_part10.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)   |   1qzv    | 


#### Download the bag-level label for training set, the training list, and the validation list.

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|bag_level_label.json |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|train.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|query.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|gallery.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


#### Download the test set

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  sysu_test_set_all.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


## Data organization

At last, the folder looks like:

```
SYSU-30k-released
├── SYSU-30k-released
│   ├── meta
│   |   ├── train.txt (for weakly supervised training, "filename\n" in each line)
│   |   ├── query.txt (for evaluation)
│   |   ├── gallery.txt (for evaluation)
│   ├── sysu_train_set_all
│   |   ├── 0000000001
│   |   ├── 0000000002
│   |   ├── 0000000003
│   |   ├── 0000000004
│   |   ├── ...
│   |   ├── 0000028309
│   |   ├── 0000028310
│   ├── sysu_test_set_all
│   |   ├── gallery
│   |   |   ├── 000028311
│   |   |   |   ├── 000028311_c1_1.jpg
│   |   |   ├── 000028312
│   |   |   |   ├── 000028312_c1_1.jpg
│   |   |   ├── 000028313Leaderboard on SYSU-30k.
│   |   |   |   ├── 000028313_c1_1.jpg
│   |   |   ├── 000028314
│   |   |   |   ├── 000028314_c1_1.jpg
│   |   |   ├── ...
│   |   |   |   ├── ...
│   |   |   ├── 000029309
│   |   |   |   ├── 000029309_c1_1.jpg
│   |   |   ├── 000029310
│   |   |   |   ├── 000029310_c1_1.jpg
│   |   |   ├── 0000others
│   |   |   |   ├── 0000others_c1_1.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── ...
│   |   ├── query
│   |   |   ├── 000028311
│   |   |   |   ├── 000028311_c2_2.jpg
│   |   |   ├── 000028312
│   |   |   |   ├── 000028312_c2_2.jpg
│   |   |   ├── 000028313
│   |   |   |   ├── 000028313_c2_2.jpg
│   |   |   ├── 000028314
│   |   |   |   ├── 000028314_c2_2.jpg
│   |   |   ├── ...
│   |   |   |   ├── ...
│   |   |   ├── 000029309
│   |   |   |   ├── 000029309_c2_2.jpg
│   |   |   ├── 000029310
│   |   |   |   ├── 000029310_c2_2.jpg
```



## Evaluation metric

We fix the train/test partitioning. In the test set, we choose 1,000 images belonging to 1,000 different person IDs to form the query set. As the scalability is vital for the practicability of Re-ID systems, we propose to challenge a Re-ID model's scalability by providing a gallery set containing a vast volume of distractors for validation. Specifically, for each probe, there is only one matching person image as the correct answer in the gallery. At the same time, there are 478,730 mismatching person images as the wrong answer in the gallery. Thus, the evaluation protocol is to search for a needle in the ocean, just like the police search a massive amount of videos for a criminal. We use the rank-1 accuracy as the evaluation metric.


## Due to the large number of mismatching person images, SYSU-30k test set is so challenge that existing methods may have poor performance on SYSU-30k test set. Therefore, no matter whether your method is trained on SYSU-30k training set, you are encouraged to evaluate your method in SYSU-30k test set. 

# For a fair evaluation, please refer to the evaluation code in Using_SYSU30k_Test_Set/test_sysu_combine.py. 


# At present, the performance of existing methods on the SYSU-30k test set is poor. Therefore, to report new results on the SYSU-30k test set is encouraged.

### Table 3: Results on SYSU-30k.
  
| Supervision      | Method       |  Rank-1 | 
|:------------------:|:------------------:|:------------------:|
| Generalization   | [DARI [1]](https://arxiv.org/abs/1604.04377)        |        11.2  |
| Generalization   |  [DF [2]](https://arxiv.org/abs/1512.03622)   |   10.3     | 
| Generalization   |  [ResNet-50 [3]](https://arxiv.org/abs/1512.03385)      |     20.1         |
| Generalization   | [Local-CNN [4]](https://dl.acm.org/doi/10.1145/3240508.3240645)       |     23.0         |
| Generalization   | [MGN [5]](https://arxiv.org/abs/1804.01438)             |     23.6         |
| Generalization   | [IICS [6]](https://arxiv.org/abs/2103.11658)         |     36.0         |
| Weakly Supervised  | [DGL [7]](https://arxiv.org/abs/1904.03845)      |     26.9         |
| Self Supervised   | [SimCLR [8]](https://arxiv.org/abs/2002.05709)          |     10.9         |
| Self Supervised   | [MoCo v2 [9]](https://arxiv.org/abs/2003.04297)        |     11.6         |
| Self Supervised   | [BYOL [10]](https://arxiv.org/abs/2006.07733)         |     12.7         |
| Self Supervised   | [Triplet [11]](https://arxiv.org/abs/2104.08760)         |     14.8         |

[1] Guangrun Wang, Liang Lin, Shengyong Ding, Ya Li, Qing Wang: DARI: Distance Metric and Representation Integration for Person Verification. AAAI 2016: 3611-3617

[2] Shengyong Ding, Liang Lin, Guangrun Wang, Hongyang Chao: Deep feature learning with relative distance comparison for person re-identification. Pattern Recognit. 48(10): 2993-3003 (2015)

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

[4] Jiwei Yang, Xu Shen, Xinmei Tian, Houqiang Li, Jianqiang Huang, Xian-Sheng Hua: Local Convolutional Neural Networks for Person Re-Identification. ACM Multimedia 2018: 1074-1082

[5] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou: Learning Discriminative Features with Multiple Granularities for Person Re-Identification. ACM Multimedia 2018: 274-282

[6] Shiyu Xuan, Shiliang Zhang: Intra-Inter Camera Similarity for Unsupervised Person Re-Identification. CVPR 2021: 11926-11935

[7] Guangrun Wang, Guangcong Wang, Xujie Zhang, Jianhuang Lai, Zhengtao Yu, Liang Lin: Weakly Supervised Person Re-ID: Differentiable Graphical Learning and a New Benchmark. IEEE Trans. Neural Networks Learn. Syst. 32(5): 2142-2156 (2021)

[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey E. Hinton: A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020: 1597-1607

[9] Xinlei Chen, Haoqi Fan, Ross B. Girshick, Kaiming He: Improved Baselines with Momentum Contrastive Learning. CoRR abs/2003.04297 (2020)

[10] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Ávila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning. NeurIPS 2020

[11] Guangrun Wang, Keze Wang, Guangcong Wang, Philip H.S. Torr, and Liang Lin, "Solving Inefficiency of Self-supervised Representation Learning" [C]. in Proc. of IEEE International Conference on Computer Vision (ICCV), 2021.


# Pretrained models


### Download the pretrained model

 | Pretrained      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  ResNet50-sysu30k-2048-AsFeature/net_6.pth    |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


### Requirements

We have tested the following versions of OS and softwares:

- Python 3.6+
- PyTorch 1.1 or higher
- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1


### Test with pretrained model

Evaluating a trained model includes two steps, i.e., feature extraction (which is in fact classificaction probability vector) and metric score caculation.


```shell
**Step 1**: python test_sysu.py --gpu_ids ${GPU_ID} --name ${NAME_OF_MODEL} --test_dir ${DIR_OF_TEST_SET}  --which_epoch ${WHICH_EPOCH_OF_CHECKPOINT} --batchsize ${BATCH_SIZE}
**Step 2**: python evaluate_sysu.py
```
Arguments are:
- `--gpu_ids ${GPU_ID}`: the gpu IDs you use.
- `--name ${NAME_OF_MODEL}`: the name of the model, which is also the dir of the saved checkpoints and logs 
- `--test_dir ${DIR_OF_TEST_SET}`: the dir of your test set
- `--which_epoch ${WHICH_EPOCH_OF_CHECKPOINT}`: which epoch of the checkpoint you want to evaluate
- `--batchsize ${BATCH_SIZE}`: the batch size for testing

An example:
```shell
cd SYSU-30k/GraphReID/
python test_sysu.py  --gpu_ids 0  --name work_dirs/ResNet50-sysu30k-2048-AsFeature  --test_dir  /data1/wangguangrun/sysu_test_set_all/    --which_epoch 6  --batchsize 100
python evaluate_sysu.py
```

**Note**: Due the huge consumption of hard disks, sometimes, the above two steps can be combined into one step, e.g.:
```shell
cd SYSU-30k/GraphReID/
python test_sysu_combine.py  --gpu_ids 0  --name work_dirs/ResNet50-sysu30k-2048-AsFeature  --test_dir  /data1/wangguangrun/sysu_test_set_all/    --which_epoch 6  --batchsize 100
```


### Training a model in a weakly supervised manner

**Note**: During training, checkpoints and logs are saved the folder named "work_dir", which will occupies much memory of the hard disk. If you want to save your weights somewhere else, please use symlink, for example:

```shell
cd GraphReID
ln -s /data1/wangguangrun/GraphReID  work_dirs
```

**Training the full model On Market-1501** (including unary term, pairwise term, and weakly-supervised triplet loss):

```shell
cd GraphReID
CUDA_VISIBLE_DEVICES=1,2  python3 main.py --datadir /data1/wangguangrun/dataset/market1501/ --bagid 2 --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+2*Triplet   --margin 1.2 --save adam_weak_market           --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad 
```


**Training with with only unary term On Market-1501**:

```shell
cd GraphReID
python train_weak.py --gpu_ids 2 --name ResNet50 --color_jitter  --train_all --batchsize 32 --erasing_p 0.5 --data_dir /home/wangguangrun/weakly-reid/pytorch 
```


# Agreement
* The SYSU-30k dataset is available for non-commercial research purposes only.
* You agree **NOT** to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
* You agree **NOT** to further copy, publish or distribute any portion of the SYSU-30k dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.
* The SYSU reserves the right to terminate your access to the SYSU-30k dataset at any time.
* The person identities are released upon request for research purposes only. Please contact us for details.




# Citation

If you use these models in your research, please kindly cite:

```
@inproceedings{Wang2020Weakly_tnnls,
      title={Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark},
      author={Guangrun Wang and
              Guangcong Wang and
              Xujie Zhang and
              Jianhuang Lai and
              Zhengtao Yu and
              Liang Lin},
      booktitle={ IEEE Transactions on Neural Networks and Learning Systems (T-NNLS)},
      year={2020}
      }
```
