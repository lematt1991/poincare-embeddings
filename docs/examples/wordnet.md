---
id: wordnet
title: Reproducing Wordnet Reconstruction Results
sidebar_label: Reproducing Wordnet Reconstruction Results
---

## Generate Data

First generate the transitive closure of wordnet data using the following command:

```Bash
cd wordnet
python transitive_closure.py
```

This will generate the transitive closure of the full noun hierarchy as well as of the mammals subtree of WordNet.

To embed the mammals subtree in the reconstruction setting (i.e., without missing data), go to the root directory of the project and run

```Bash
./train-mammals.sh
```	


This shell script includes the appropriate parameter settings for the mammals subtree and saves the trained model as `mammals.pth`.