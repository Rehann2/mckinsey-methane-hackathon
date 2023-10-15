# Directory to store bash-scripts to execute pytorch training pipelines.

From the root directory:

1. First, you need to make bash-script executable: `chmod u+x ./bash/convnext-base-train.sh`
   
2. Execute: `./bash/convnext-base-train.sh`


# Training Keras models from the Terminal
- From the main directory, run: `python train_keras.py <name of the model>` from the command line interface.
- Running this model will create a log file to track the progress and see previous model runs. To access the log files and check the details about the fitted model, run  `tensorboard --logdir=logs` from the command line interface.
