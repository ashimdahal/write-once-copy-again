# Whats this?
Write once copy again is my personal template from to train, test and save any deep learning models and projects from now on. In it's current state it can handle running on a multinode multigpu setting with appropriate multiprocessing. More multi-multi-cool things comming soon.

ddp_training provides a complete Trainer class that can be used to train any model on a multinode ddp setting.

non_ddp_training provides a complete Trainer class for single gpu systems.

## features
1. save and load models
2. Save and load history
3. Continue model training in checkpoint fashion
4. Able to handle DDP in multinode systems like an HPC

## todo
[] argparse inception: add argparse to hpc.sh file to be more abstract on training
[] add write once template for testing


