# SYSU-30k dataset, code, and pretrained model

The Dataset, code, and pretrained model of "Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark" https://arxiv.org/abs/1904.03845.


The source code of our weakly supervised re-ID is originally written by [**Guangcong Wang**](https://wanggcong.github.io) who has rich experiences in person re-ID, and is partially revised by [Guangrun Wang](https://wanggrun.github.io/).


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

The test set is uncompressed due to the appropriate memory size.

### Download the training set

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


### Download the bag-level label for training set, the training list, and the validation list.

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|bag_level_label.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|train.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 
|val.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


### Download the test set

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  sysu_test_set_all.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


## Data organization

At last, the folder looks like:

```
SYSU-30k-released
├── imagenet
│   ├── meta
│   |   ├── train.txt (for weakly supervised training, "filename\n" in each line)
│   |   ├── val.txt (for evaluation)
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
│   |   |   ├── 000028313
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

We fix the train/test partitioning. In the test set, we choose 1,000 images belonging to 1,000 different person IDs to form the query set. As the scalability is important for the practicability of Re-ID systems, we propose to challenge the scalability of a Re-ID model by providing a gallery set containing a vast volume of distractors for validation. Specifically, for each probe, there is only one matching person image as the correct answer in the gallery, while there are 478,730 mismatching person images as the wrong answer in the gallery. Thus, the evaluation protocol is to search for a needle in the ocean, just like the police search a massive amount of videos for a criminal. We use the rank-1 accuracy as the evaluation metric.


# Pretrained models


### Download the pretrained model

 | Pretrained      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  pretrained_model.pth      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


### Requirements

We have tested the following versions of OS and softwares:

- Python 3.6+
- PyTorch 1.1 or higher
- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1


### Test with pretrained model

Evaluating a trained model includes two steps, i.e., feature extraction (which is in fact classificaction probability vector) and metric score caculation.


```shell
**Step 1**: python test_sysu_pretrained.py --gpu_ids ${GPU_ID} --name ${NAME_OF_MODEL} --test_dir ${DIR_OF_TEST_SET}  --which_epoch ${WHICH_EPOCH_OF_CHECKPOINT} --batchsize ${BATCH_SIZE}
**Step 2**: python evaluate_sysu_pretrained.py
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_sysu_pretrained.py  --gpu_ids 0  --name model/ResNet50-sysu30k-2048-AsFeature  --test_dir  /data1/wangguangrun/sysu_test_set_all/    --which_epoch 6  --batchsize 100
python evaluate_sysu_pretrained.py
```

**Note**: Due the huge consumption of hard disks, sometimes, the above two steps can be combined into one step, e.g.:
```shell
cd SYSU-30k/GraphReID/
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_sysu_pretrained_combine.py  --gpu_ids 0  --name model/ResNet50-sysu30k-2048-AsFeature  --test_dir  /data1/wangguangrun/sysu_test_set_all/    --which_epoch 6  --batchsize 100
```


### Training a model in a weakly supervised manner

**Note**: During training, checkpoints and logs are saved the folder named "work_dir", which will occupies much memory of the hard disk. If you want to save your weights somewhere else, please use symlink, for example:

```shell
cd GraphReID
ln -s /data1/wangguangrun/GraphReID  work_dirs
```






# Citation

If you use these models in your research, please cite:

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
