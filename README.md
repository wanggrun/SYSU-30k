# SYSU-30k dataset, code, and pretrained model

SYSU-30k Dataset of "Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark" https://arxiv.org/abs/1904.03845


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

**Note** that our original training set occupies 462G's memory. We are not able to upload the original data taking up such a large memory. As a result, we downsample the train images from 288 * x resolution to 144 * x resolution with x representing the shortest edge. The compressed data sum up to 100+G.

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
|  sysu_train_set_all_part10.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


### Download the bag-level label for training set

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|bag_level_label.txt |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


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
│   ├── train
│   ├── val
```



## Evaluation metric

We fix the train/test partitioning. In the test set, we choose 1,000 images belonging to 1,000 different person IDs to form the query set. As the scalability is important for the practicability of Re-ID systems, we propose to challenge the scalability of a Re-ID model by providing a gallery set containing a vast volume of distractors for validation. Specifically, for each probe, there is only one matching person image as the correct answer in the gallery, while there are 478,730 mismatching person images as the wrong answer in the gallery. Thus, the evaluation protocol is to search for a needle in the ocean, just like the police search a massive amount of videos for a criminal. We use the rank-1 accuracy as the evaluation metric.


# Pretrained models

The source code of our weakly supervised re-ID is originally written by [Guangcong Wang](https://wanggcong.github.io) who has rich experiences in re-ID, and is partially revised by [Guangrun Wang](https://wanggrun.github.io/).


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
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9/5.3/5.4/7.3


### Test with pretrained model

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
Optional arguments are:
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--pretrained ${PRETRAIN_WEIGHTS}`: Load pretrained weights for the backbone.
- `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

An example:
```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/xxx/
bash tools/dist_train.sh config/xx.py 8
```
**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s /DATA/}/work_dirs
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
