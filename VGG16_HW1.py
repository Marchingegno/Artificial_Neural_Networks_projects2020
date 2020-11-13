import os
from datetime import date, datetime

import tensorflow as tf
from keras.applications import vgg16
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import callbacks

training_dir = "training_sorted"
valid_dir = "validation_sorted"
log_dir = "logs"
checkpoints_dir = "checkpoint_callback"

seed = 1234

tf.random.set_seed(seed)
apply_data_augmentation = False

# ImageDataGenerator
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=10,
        height_shift_range=10,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="categorical",
        cval=0,
        rescale=1. / 255
    )
else:
    train_data_gen = ImageDataGenerator(
        # preprocessing_function=vgg16.preprocess_input
        rescale=1. / 255
    )

target_size = (256, 256)  # 224 are VGG16 imagenet default size
batch_size = 32

# Generators
train_gen = train_data_gen.flow_from_directory(
    directory=training_dir,
    target_size=target_size,
    # classes=["NOMASK", "ALLMASK", "SOMEMASK"],
    color_mode="rgb",
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    class_mode="categorical"
)

valid_gen = train_data_gen.flow_from_directory(
    directory=valid_dir,
    target_size=target_size,
    # classes=["NOMASK", "ALLMASK", "SOMEMASK"],
    color_mode="rgb",
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    class_mode="categorical"
)

# Datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, target_size[0], target_size[1], 3], [None, 3])
)

train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_generator(
    lambda: valid_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, target_size[0], target_size[1], 3], [None, 3])
)

valid_dataset = valid_dataset.repeat()

# Model
vgg = vgg16.VGG16(
    include_top=False,
    weights="imagenet",
    classes=3,
    input_shape=[target_size[0], target_size[1], 3]
)
enable_finetuning = True

if enable_finetuning:
    freeze_until = 15  # Layer from which we want to fine-tune.
    for layer in vgg.layers[:freeze_until]:
        layer.trainable = False
else:
    vgg.trainable = False

model = tf.keras.Sequential()
model.add(vgg)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation="relu"))
model.add(tf.keras.layers.Dense(units=3, activation="softmax"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

print(model.summary())

# Define callbacks

callback_list = []
earlyStopping = True
modelCheckpoint = True
tensorBoard = False

if earlyStopping:
    es = callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=10,
        verbose=False,
        mode='auto',
        baseline=None,
        restore_best_weights=True)
    callback_list.append(es)

if modelCheckpoint:
    filepath = os.path.join(checkpoints_dir, "day{0}_at_{1}".format(date.today().strftime("%d_%b"), datetime.now().strftime("%H_%M")) + "__checkpoint-epoch={epoch:02d}-valacc={val_accuracy:.2f}.hdf5")

    cp = callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_accuracy',
        verbose=True,
        save_best_only=False,
        save_weights_only=True,
        mode='auto',
        save_freq="epoch")
    callback_list.append(cp)

if tensorBoard:
    tb = callbacks.TensorBoard(
        log_dir=log_dir,
        profile_batch=0,
        histogram_freq=1)  # if 1 shows weights histograms.
    callback_list.append(tb)

model.fit(
    x=train_dataset,
    callbacks=callback_list,
    epochs=2,
    steps_per_epoch=len(train_gen),
    validation_data=valid_dataset,
    validation_steps=len(valid_gen)
)
