

### Download the test set

 | Dataset      | Link to download       |  baidu pan code | 
|:------------------:|:------------------:|:------------------:|
|  sysu_test_set_all.tar      |  [:arrow_down:](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ)    |   1qzv    | 


## Data organization

At last, the folder looks like:

The folder looks like:

```
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

We fix the train/test partitioning. In the test set, we choose 1,000 images belonging to 1,000 different person IDs to form the query set. As the scalability is vital for the practicability of Re-ID systems, we propose to challenge a Re-ID model's scalability by providing a gallery set containing a vast volume of distractors for validation. Specifically, for each probe, there is only one matching person image as the correct answer in the gallery. At the same time, there are 478,730 mismatching person images as the wrong answer in the gallery. Thus, the evaluation protocol is to search for a needle in the ocean, just like the police search a massive amount of videos for a criminal. We use the rank-1 accuracy as the evaluation metric.



# For a fair evaluation, please refer to the evaluation code in Using_SYSU30k_Test_Set
