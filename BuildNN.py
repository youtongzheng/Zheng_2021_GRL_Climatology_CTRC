import sys, subprocess
out = subprocess.getoutput('/bin/tcsh -c "module load tensorflow && printenv PYTHONPATH"')
sys.path = out.split("\n")[-1].split(":") + sys.path
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([keras.layers.Flatten(input_shape=(25,)),
                         keras.layers.Dense(256, activation="relu"),
                         keras.layers.Dense(256, activation="relu"),
                         keras.layers.Dense(5)
                         ])
model.compile('adam', 'mse',metrics=["accuracy"])
model.summary()