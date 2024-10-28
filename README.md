# GELRTLM-A-Semi-Supervised-Multi-SensorS-Information-Fusion-Tensor-Learning-For-Fault-Diagnosis
This is the code for the paper entitled "Semi-supervised multi-sensor information fusion tailored graph embedded low-rank tensor learning machine under extremely low labeled rate" [[paper source]](https://doi.org/10.1016/j.inffus.2023.102222), published in the Information Fusion,  January 2024.<br>
First Author: Haifeng Xu.<br>
Organization: Department of Mechanical Engineering, Tsinghua University, Beijing, 100084, China;<br>
School of Mechanical Engineering, Anhui University of Technology, Ma’anshan, 243032, China.

# Citation
For further introductions to semi-supervised tensor learning and information fusion in bearing fault diagnosis, please read our [paper](https://doi.org/10.1016/j.inffus.2023.102222). And if you find this repository useful and use it in your works, please cite our paper, thank you~:  <br>

@article{XU2024102222,<br>
author = {Haifeng Xu and Xu Wang and Jinfeng Huang and Feibin Zhang and Fulei Chu},<br>
title = {Semi-supervised multi-sensor information fusion tailored graph embedded low-rank tensor learning machine under extremely low labeled rate},<br>
journal = {Information Fusion},<br>
volume = {105},<br>
pages = {102222},<br>
year = {2024},<br>
issn = {1566-2535},<br>
doi = {10.1016/j.inffus.2023.102222},  <br>
abstract = {This paper investigates a demanding and meaningful task of intelligent fault diagnosis, in which multi-sensors signals are fused for semi-supervised analysis with few labeled fault data. Exploring effective strategies to solve this task from an industrial or academic perspective remains a challenging and resource-consuming task. To this issue, this study develops a new low-rank tensor-based semi-supervised classifier, called graph embedded low-rank tensor learning machine (GE-LRTLM), which can effectively alleviate the foregoing difficulties and increase the diagnosis precision in engineering applications. First, multi-sensor and multi-channel vibration signals are converted into the pixel matrices that are stacked as the multi-sensor information fusion feature tensors, making the coupling relationship between multi-sensor signals remained and achieving a reasonable fusion of multi-source features. Additionally, an advanced tensor decomposition method, tensor nuclear norm (TNN), is introduced in GE-LRTLM model to obtain the low-rank structure information of each feature tensor, implementing the extraction of the most important feature and patterns from the tensor data while ensuring that the tensor data structure remains unchanged. Ultimately, the manifold regularization and tensor-based graph construction method are introduced to obtain potential label information from unlabeled samples, achieving better description of the geometry similarity and distribution of tensor data. Numerous semi-supervised experiments are conducted across multiple datasets, and the results demonstrate that the proposed method can achieve a classification accuracy of 97% even when the number of labeled samples is extremely limited. Simultaneously, it also verifies that the combination of the constructed labeled and unlabeled multi-sensor information fusion tensor samples can promote the improvement of model accuracy.}
}
