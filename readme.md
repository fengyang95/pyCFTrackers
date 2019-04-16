# pyCFTrackers
Python re-implementation of some correlation filter based tracker, and all of these algorithms are implemented 
based on the official Matlab code. All the code has been tested on Ubuntu 16.04, Python 3.5.
I use [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to eval the performance on OTB and VOT. 

- [x] [MOSSE](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.294.4992&rep=rep1&type=pdf)
- [x] [CSK](http://59.80.44.48/www.robots.ox.ac.uk/~joao/publications/henriques_eccv2012.pdf)
- [x] [CN](http://117.128.6.12/cache/www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/CN_Tracking_CVPR14.pdf?ich_args2=465-31142901008185_f9df5d61efad793a151f3e0f467d3f75_10001002_9c896128d7c2f2d6933d518939a83798_91ccc5b03febd95ae516eb0f69b18b49)
- [x] [KCF/DCF](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)
- [x] [DSST](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/ScaleTracking_BMVC14.pdf)
- [x] [Staple](https://arxiv.org/pdf/1512.01355v2.pdf)
- [x] [BACF](http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Learning_Background-Aware_Correlation_ICCV_2017_paper.pdf)  
- [x] [CSR-DCF](https://arxiv.org/pdf/1611.08461v1.pdf)   
- [x] [DAT](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/dat/)  This tracker is not based on CF, I re-implement it just for understanding Staple better  


**Trackers from other repos**
- [x] [ECO/ECO-HC](https://arxiv.org/pdf/1611.09224v1.pdf) from [pyECO](https://github.com/StrangerZhang/pyECO)
- [x] [OPENCV_CSRDCF](https://github.com/opencv/opencv_contrib)(Note that trackers implemented in opencv get lower score because they will return a success flag and if this flag is False, the bbox will be set to (0,0,0,0))


## Install
``` bash
git clone https://github.com/wwdguu/pyCFTrackers.git && cd pyCFTrackers
export pyCFTrackers=$PWD

pip install -r requirements.txt

cd lib/eco/features/
python setup.py build_ext --inplace
cd ../../..

cd lib/pysot/utils/
python setup.py build_ext --inplace
cd ../../..

export PYTHONPATH=$PWD:$PYTHONPATH
```

## Get Dataset
You can follow the instructions in the following repo to get VOT2016,VOT2018 and OTB100 dataset.
[trackdat](https://github.com/jvlmdr/trackdat/tree/master/python/trackdat)  
Then get the json files according to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)
Then put the data in the dataset dir.

## Demo
``` bash
cd examples
python cf_demo.py
```
![demo](results/Coke_vis.gif)
## Eval on OTB and VOT
``` bash
cd eval
python get_vot2016_result.py
python get_vot2018_result.py
python ope_otb.py
python eval_VOT2016.py
python  eval_VOT2018.py
python eval_OTB.py
```

## OTB result
### OTB-100
|     OTB100 Success Plot   	    | OTB100 Precision Plot	    |
| --------------------------------- | ----------------------------- |
|![](results/pytracker_OPE_OTB100_success.png)  	    |![](results/pytracker_OPE_OTB100_precision.png)  	    |


### OTB-2013
|     OTB2013 Success Plot   	    | OTB2013 Precision Plot	    |
| --------------------------------- | ----------------------------- |
|![](results/pytracker_OPE_OTB2013_success.png)  	    |![](results/pytracker_OPE_OTB2013_precision.png)  	    |


## VOT result


```
VOT2018
---------------------------------------------------------------
| Tracker Name  | Accuracy | Robustness | Lost Number |  EAO  |
---------------------------------------------------------------
|      ECO      |  0.485   |   0.403    |    86.0     | 0.224 |
|    CSRDCF     |  0.497   |   0.478    |    102.0    | 0.212 |
|    ECO-HC     |  0.500   |   0.473    |    101.0    | 0.207 |
| OPENCV_CSRDCF |  0.486   |   0.651    |    139.0    | 0.170 |
|    Staple     |  0.528   |   0.707    |    151.0    | 0.166 |
|      DAT      |  0.477   |   0.777    |    166.0    | 0.158 |
|     BACF      |  0.508   |   1.054    |    225.0    | 0.119 |
|      CN       |  0.439   |   1.100    |    235.0    | 0.112 |
|     DSST      |  0.503   |   1.288    |    275.0    | 0.101 |
|      DCF      |  0.463   |   1.246    |    266.0    | 0.099 |
|      KCF      |  0.463   |   1.330    |    284.0    | 0.094 |
|      CSK      |  0.418   |   1.386    |    296.0    | 0.090 |
|     MOSSE     |  0.380   |   1.868    |    399.0    | 0.069 |
---------------------------------------------------------------

VOT2016
---------------------------------------------------------------
| Tracker Name  | Accuracy | Robustness | Lost Number |  EAO  |
---------------------------------------------------------------
|      ECO      |  0.564   |   0.256    |    55.0     | 0.336 |
|    ECO-HC     |  0.532   |   0.350    |    75.0     | 0.293 |
|    CSRDCF     |  0.537   |   0.326    |    70.0     | 0.284 |
|    Staple     |  0.557   |   0.382    |    82.0     | 0.280 |
| OPENCV_CSRDCF |  0.521   |   0.438    |    94.0     | 0.239 |
|      DAT      |  0.474   |   0.503    |    108.0    | 0.232 |
|     BACF      |  0.532   |   0.592    |    127.0    | 0.196 |
|      CN       |  0.468   |   0.653    |    140.0    | 0.178 |
|     DSST      |  0.540   |   0.736    |    158.0    | 0.175 |
|      DCF      |  0.474   |   0.704    |    151.0    | 0.171 |
|      KCF      |  0.469   |   0.718    |    154.0    | 0.171 |
|      CSK      |  0.433   |   0.886    |    190.0    | 0.139 |
|     MOSSE     |  0.389   |   1.258    |    270.0    | 0.094 |
---------------------------------------------------------------

```
## License
Licensed under an MIT license.



