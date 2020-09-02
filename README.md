# SYSU-30k
SYSU-30k Dataset of "Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark" https://arxiv.org/abs/1904.03845


## Statistic of the dataset

 SYSU-30k contains 30k categories of persons, which is about 20 times large rthan CUHK03 (1.3k categories)and Market1501 (1.5k categories), and 30 times larger than ImageNet (1k categories). SYSU-30k contains 29,606,918 images. Moreover, SYSU-30k provides not only a large platform for the weakly supervised ReID problem but also a more challenging test set that is consistent with the realistic setting for standard evaluation. Figure 1 shows some samples from the SYSU-30k dataset. 
 

| Dataset      | CUHK03       |  Market-1501 |   Duke       |      MSMT17  |       CUHK01 |         PRID |        VIPeR |       CAVIAR |      SYSU-30k|
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| Categories   | 1,467        | 1,501.       |   1,812      |      4,101   |        971    |        934  |        632   |        72    |      30,508  |
|   Scene      |    Indoor    |     Outdoor  |   Outdoor   |Indoor, Outdoor|    Indoor     |   Outdoor   |    Outdoor   |   Indoor     |Indoor, Outdoor|
|   Annotation |    Strong    |     Strong   |   Strong     |  Strong      |    Strong     |  Strong     |   Strong     |    Strong    |  Weak         |




<p align="center">
<img src = "https://github.com/wanggrun/SYSU-30k/blob/master/sysu-30k-statistics.png", width='300'>
 </p>

<p align='center'>Figure 2: The statistics of the SYSU-30k. (a) summarizes the number of the bags with respect to the number of the images per bag. (b) and (c) compare SYSU-30k with the existing datasets in terms of image number and person IDs for both the entire dataset and the test set.</p>


## Visualization of the dataset

<p align="center">
<img src="https://github.com/wanggrun/SYSU-30k/blob/master/sysu-30k-example.png", width = '400'>
 </p>

 <p align='center'>Figure 1: Examples in our SYSU-30k dataset. (a) training images in terms of bag; (b) their bag-level annotations; (c) test set.</p>
 
 

## Download the dataset


## Data organization


## Evaluation metric


# Pretrained models


# Citation

If you use these models in your research, please cite:

@inproceedings{wang2020weakly,
  
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
