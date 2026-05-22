# Functional Connectivity as an Architectural Prior for EEG Decoding Models

What is in this repository?
-> The interface you will interact and train the model with : all_in_one_ftc.py
-> The testing script once you have the weights : test_evaluator.py
-> The actual engine : /model/TmpEncoder___.py
-> The preprocessing scripts : Sourced from CBraMod
-> The dataloading scripts : Sourced from CSBrain

There are two types of engines. Continuous and Random/Modulo as described in the paper. The details of the most important, changeable flags used are as follows:

1) --nc : Number of classes in your dataset.
2) --mode : The mode of training. Is your dataset binary/multiclassification? Or regression?
3) --config : The engine that you want to use. They are both different engines because RegionAttention tensors are different.
4) --dataset : The dataset you want to use.
5) --state_dict_path : The path where your trained state dict is stored
6) --dataset_dir : The path where your dataset db or files are stored. This depends. Look at preprocessing code to learn more about file structure.
7) --avail_gpus : Which GPU you want to train on. By default, it supports only 1 GPU, but can be extended to multiple by using DataParallel or DDP using a few lines.

A sample command for the all_in_one_ftc.py looks as follows: python3 all_in_one_ftc.py --batch_size 64 --dataset_dir '~/simpletmp/data/FACED/db' --mode 'mul' --dataset 'FACED' --config 'random' --cuda 1 --avail_gpus '1' --lr 5e-4 -- epochs 40 --clip_value 7.5

A sample command for the test evaluator looks as follows: python3 test_eval.py --batch_size 8 --dataset_dir '~/simpletmp/data/FACED/db' --mode 'mul' --dataset 'FACED' --config 'random' --state_dict_path '~/simpletmp/saved_fm/finetune_weights_17_faced_42_cont.pth'

Huge thanks to open-source programs CBraMod and CSBRain !
