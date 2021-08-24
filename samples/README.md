# TFRD

> This is the dataset for temperature field reconstruction of heat source systems task (TFR-HSS).
>
> author: Zhiqiang Gong
>
> contact: gongzhiqiang13@nudt.edu.cn
>
> time: 2021-08-24

## Structures

> TFRD contains three sub-data: namely the HSink, the ADlet, and the DSine corresponding to the three typical sub-task.

* TFRD  // The general list of TFRD

  * HSink  // Data for HSink sub-task

    * train  // Data for training

      > The training process read the training samples from `train_val.txt` and find the corresponding sample in `train/`

      * train  // training samples
      * train_val.txt  // sample list for training 

    * test

      > Each test consists six test sets:
      >
      > `test 0` represents the general test samples
      >
      > `test 1` describes the test set where all the components share the same power density
      >
      > `test 2` describes the test set where 1/4 of the heat sources are with zero-power intensity and the remainder are with random selected power intensity.
      >
      > `test 3` describes the test set where half of the heat sources are with zero-power intensity and half with random selected power intensity.
      >
      > `test 4` describes the test set where 3/4 of the heat sources are with zero-power intensity and the remainder are with random selected power intensity.
      >
      > `test 5` describes the test set where only one heat source is with random selected intensity and the remainder are with zero-power intensity.

      * test_0  //  test 0
      * test_0.txt // sample list for test 0
      * test_1  // special test 1 
      * test_1.txt  // sample list for special test 1
      * test_2  // special test samples 2 
      * test_2.txt  // sample list for special test 2
      * test_3  // special test samples 3 
      * test_3.txt  // sample list for special test 3
      * test_4  // special test samples 4 
      * test_4.txt  // sample list for special test 4
      * test_5  // special test samples 5 
      * test_5.txt  // sample list for special test 5

  * ADlet  // Data for ADlet sub-task

    * train
      * train
      * train_val.txt
    * test
      * test_0
      * test_0.txt
      * test_1
      * test_1.txt
      * test_2
      * test_2.txt
      * test_3
      * test_3.txt
      * test_4
      * test_4.txt
      * test_5
      * test_5.txt

  * DSine  // Data for DSine sub-task

    * train
      * train
      * train_val.txt
    * test
      * test_0
      * test_0.txt
      * test_1
      * test_1.txt
      * test_2
      * test_2.txt
      * test_3
      * test_3.txt
      * test_4
      * test_4.txt
      * test_5
      * test_5.txt

## Variables

* `F`: The component information, including the power density, position and shape
* `u`: Real temperature field of the specific `F` in `200*200` discretized matrix
* `u_obs`: Temperature value of the monitoring points
* u_pos: Positions of monitoring points in `200*200` discretized matrix (1 describes area monitoring points and 0 without monitoring points)
* `xs`, `ys`, `zs`: Corresponding coordinates of the `200*200` discretized matrix

## Examples

* General examples

| ![HSink](https://i.loli.net/2021/08/24/l3XKCiBR2qrok4x.png) | ![ADlet](https://i.loli.net/2021/08/24/jX4LygbmIJoxFcU.png) | ![DSine](https://i.loli.net/2021/08/24/UOtGxEnwXfAWp8S.png) |
| :---------------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
|                            HSink                            |                            ADlet                            |                            DSine                            |

* Special examples of HSink

| ![test0](https://i.loli.net/2021/08/24/bJczHyBj7xq28hE.png) | ![test1](https://i.loli.net/2021/08/24/k5paVYfB3D9CTbn.png) | ![test2](https://i.loli.net/2021/08/24/VMRChBWpibl5kIc.png) |
| :---------------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
|                           Test 0                            |                           Test 1                            |                           Test 2                            |
| ![test3](https://i.loli.net/2021/08/24/1XOeAG5dPqinaHQ.png) | ![test4](https://i.loli.net/2021/08/24/89yAIOEWMtx4UYc.png) | ![test5](https://i.loli.net/2021/08/24/YOVdCjxMBUuk1pe.png) |
|                           Test 3                            |                           Test 4                            |                           Test 5                            |

## Citing this work

If this dataset is helpful for your research, please  consider citing:

```
@article{gong2021,
    Author = {Xiaoqian Chen and Zhiqiang Gong and Xiaoyu Zhao and Wen Yao},
    Title = {TFRD: A Benchmark Dataset for Research on Temperature Field Reconstruction of Heat-Source Systems},
    Journal = {arXiv preprint arXiv:2108.08298},
    Year = {2021}
}
```