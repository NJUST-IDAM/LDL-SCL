# LDL-SCL

Introduction
--
Label distribution learning (LDL) can be viewed as the generalization of multi-label learning. This novel paradigm focuses
on the relative importance of different labels to a particular instance.

Publication
--
Code accompanying papers 

**Label distribution learning by exploiting sample correlations locally**. AAAI 2018.
https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16664

**Distribution Learning with Label Correlations on Local Samples**. TKDE 2019.
https://ieeexplore.ieee.org/document/8847453

DataSet
--
The group of PALM provides some LDL data sets. http://palm.seu.edu.cn/xgeng/LDL/index.htm

How to use
--
algs.gd_ldl_scl.py:
The algorithm was proposed in paper, i.e., "Xiang Zheng, Xiuyi Jia, and Weiwei Li. Label distribution learning by exploiting sample correlations locally. In: AAAI Conference on Artificial Intelligence, New Orleans, LA, USA, 2018, pp. 4556–4563."

algs.sgd_adam_ldl_scl.py:
The extension version of the above paper, i.e., "Xiuyi Jia, Zechao Li, Xiang Zheng, Weiwei Li, Sheng-Jun Huang. Label Distribution Learning with Label Correlations on Local Samples. In: IEEE Transactions on Knowledge and Data Engineering, 2019", and it extends the algorithm with the Adam algorithm. 

algs.sgd_amsgrad_ldl_scl.py:
The extension version of the above paper, and it extends the algorithm with the amsGrad algorithm. algs.edl.py and algs.lld_bfgs.py are two algorithms compared with our methods, which are implemented with python.

**Compared with the original algorithms mentioned in the papers, the code has been optimized in convergence speed.**

Environment
--
Ubuntu 18.04

PyCharm 2018

Intel® Core™ i5-6500 CPU @ 3.20GHz × 4
