
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
```



