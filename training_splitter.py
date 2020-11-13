import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import os

training_dir = "training"
training_sorted_dir = "training_sorted"
validation_sorted_dir = "validation_sorted"
log_dir = "\\logs"
checkpoints_dir = "\\checkpoints"

tf.random.set_seed(1234)

json = pd.read_json(lines=True, path_or_buf="train_gt.json")
df = pd.DataFrame(json)
df.rename(index={0: "label"}, inplace=True)
df = df.T
df["file"] = df.index.astype(str)
df["label"] = df["label"].astype(str)
j = 0
# for i in df.index:
#    df = df.rename(index={i: str(j)})
#    j += 1
#df = df.sample(frac=1)
validation_tresh = np.ceil(0.8 * len(df))
print(validation_tresh)

# DO NOT UNCOMMENT THIS UNLESS YOU WANT TO SPLIT THE FILES AGAIN
# for i in range(0, len(df)):
#     # check if the file is to be put in validatiion or training set
#     file_path = os.path.join(training_dir, df["file"][i])
#     if i < validation_tresh:
#         # Put this in training set
#         dest_path = os.path.join(training_sorted_dir, df["label"][i])
#         dest_path = os.path.join(dest_path, df["file"][i])
#         shutil.copyfile(file_path, dest_path)
#     else:
#         # Put this in validation set
#         dest_path = os.path.join(validation_sorted_dir, df["label"][i])
#         dest_path = os.path.join(dest_path, df["file"][i])
#         shutil.copyfile(file_path, dest_path)
