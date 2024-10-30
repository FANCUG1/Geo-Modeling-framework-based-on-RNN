**Article Information**

Title: Automatic reconstruction of 3D geological models based on Recurrent Neural Network and predictive learning

Author list: Wenyao Fan, Leonardo Azevedo, Gang Liu, Qiyu Chen, Xuechao Wu, Yang Li

Affiliation: School of Computer Science, China University of Geosciences, Wuhan 430074, China 

DER/CERENA, Instituto Superior Técnico, Universidade de Lisboa, Lisbon, Portugal

**Introduction**

A geo-modeling framework based on SpatioTemporal LSTM framework. Just like videoframe prediction, 3D geo-models can be regarded as a high-dimensional tensor composing multiple continuous 2D sections. Therefore, based on the geological sections at top layer, the remained sections at different dimensions can be predicted, and finally, a 3D geo-models can be obtained by stacking these predicted sections together.

We thank to the codes and core idea provided by Wang, Y., Wu, H., Zhang, J., Gao, Z., Wang, J., Philip, S. Y., & Long, M. (2022). Predrnn: A recurrent neural network for spatiotemporal predictive learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 2208-2225. The code herein we mainly referred is at the link: https://github.com/thuml/predrnn-pytorch

**Preparation**

Before running the program, we suggest to prepare the environment with Python 3.9, PyTorch 1.10.1, NumPy 1.23.5, openCV >= 4.5.5 and SciPy 1.9.3

For running the program, you can firstly download the project and then prepare the dataset, and these training samples will be saved as .gif in the document called Continuous_3D_Model/Checker_3D_Model, finally, you can start run.py to train the network

The original dataset including both porous media sandstone and continuous fold structrual model are given at this link.

The hyperparameter settings are included in run.py.

