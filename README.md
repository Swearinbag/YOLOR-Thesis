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
  - added confusion matrix
- config file:
  - subdivisions up to 16 (more accuracy, standard at 8)
