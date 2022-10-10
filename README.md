# YOLOR-Thesis
These are all the files used/modified during the thesis about "Deep Learning Based Printed Circuit Board Component Classification". It should be noted that no weights are saved on here, as these are kept locally. The resulting confusion matrices and such can be found on here together with more data about the runs in graphs and txt files.

- - -

The main changes are in found in the following files, followed by the changes made:

- train.py:
  -  changed checkpoint export intervals
  -  added n_epoch for tracking current epoch in test.py
- test.py:
  - added confusion matrix that saves matrix/epoch with epoch number in "/conf/" folder within run directory.
  - added visual feedback for shape of predictions/labels
- detect.py:
  - added confusion matrix (called after each epoch. Returns cumulative detections)
- config file:
  - subdivisions up to 16 (more accuracy, standard at 8)



Attempted changes:
- train.py
  - freezing layers/showing layer structure (not supported by model/wrong approach)
- - -
 
 # mAP Results (percentage in function of number of iterations)

![metrics_mAP_0 5_TP_git](https://user-images.githubusercontent.com/13786325/174066742-a56143c8-da3b-4b35-87e3-3fa96d53ab6f.png)
