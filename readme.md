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
- [x] [DAT](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/dat/) **This tracker is not based on CF, I re-implement it just for understanding Staple better**
**From other repos**
- [x] [ECO/ECO-HC](https://arxiv.org/pdf/1611.09224v1.pdf) from [pyECO](https://github.com/StrangerZhang/pyECO)



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
python get_vot_result.py
python ope_otb.py
python eval_VOT2016.py
python  eval_VOT2018.py
python eval_OTB.py
```

## OTB result

![Success Plots of OPE on OTB100](results/pytracker_OPE_OTB100_success.png) 

![Precision Plots of OPE on OTB100](results/pytracker_OPE_OTB100_precision.png)

## VOT result


```
VOT2018
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|    ECO     |  0.502   |   0.384    |    82.0     | 0.235 |
|   ECO-HC   |  0.519   |   0.613    |    131.0    | 0.175 |
|    DAT     |  0.477   |   0.777    |    166.0    | 0.158 |
|   Staple   |  0.487   |   0.876    |    187.0    | 0.136 |
|    BACF    |  0.508   |   1.025    |    219.0    | 0.127 |
|    KCF     |  0.470   |   1.278    |    273.0    | 0.102 |
|    DSST    |  0.498   |   1.306    |    279.0    | 0.100 |
|    DCF     |  0.473   |   1.288    |    275.0    | 0.098 |
|     CN     |  0.421   |   1.199    |    256.0    | 0.097 |
|    CSK     |  0.418   |   1.386    |    296.0    | 0.090 |
|   MOSSE    |  0.376   |   1.999    |    427.0    | 0.061 |
------------------------------------------------------------

VOT2016
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|    ECO     |  0.560   |   0.242    |    52.0     | 0.317 |
|   Staple   |  0.519   |   0.438    |    94.0     | 0.254 |
|   ECO-HC   |  0.527   |   0.424    |    91.0     | 0.245 |
|    DAT     |  0.474   |   0.503    |    108.0    | 0.232 |
|    BACF    |  0.523   |   0.555    |    119.0    | 0.217 |
|    KCF     |  0.474   |   0.736    |    158.0    | 0.168 |
|    DSST    |  0.508   |   0.750    |    161.0    | 0.168 |
|    DCF     |  0.477   |   0.741    |    159.0    | 0.163 |
|    CSK     |  0.433   |   0.886    |    190.0    | 0.139 |
|     CN     |  0.467   |   1.203    |    258.0    | 0.117 |
|   MOSSE    |  0.394   |   1.212    |    260.0    | 0.094 |
------------------------------------------------------------
```
## License
Licensed under an MIT license.



