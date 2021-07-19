import numpy as np
import pandas as pd
import tensorflow as tf


def get_blood(num):
    switcher = {
        1: "covSR.csv",
        2: "covAE.csv",
        3: "covPM.csv"
    }
    print("dataset: "+str(switcher.get(num, "no dataset")))
    df = pd.read_csv(switcher.get(num, "no dataset"))
    x = df.sample()
    y = get_target(num, x)
    x = drop_col(num, x).to_numpy()[0]
    return x, y


def get_target(num, dataframe):
    y = int
    if num == 1:
        y = dataframe['target'].values[0]
    if num == 2:
        y = dataframe['Status'].values[0]
    if num == 3:
        y = dataframe['Status'].values[0]
    return y


def drop_col(num, dataframe):
    if num == 1:
        dataframe = dataframe.drop(columns='target')
    if num == 2:
        dataframe = dataframe.drop(columns='Status')
    if num == 3:
        dataframe = dataframe.drop(columns=['Status', 'PASIEN_ID'])
    return dataframe


def model_loader(num):
    switcher = {
        1: "San_Raphael.h5",
        2: "Albert_Einstein.h5",
        3: "Pasar_Minggu.h5"
    }
    print("model: "+str(switcher.get(num, "no model")))
    return switcher.get(num, "no model")


def get_prediction(model_file, blood_input):
    model = tf.keras.models.load_model(model_loader(model_file),
                                       custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    arr = np.array(blood_input).reshape(1, len(blood_input))
    return model.predict(arr)


if __name__ == '__main__':
    while True:
        print("#########################################################")
        pretrained = int(input("Select pre-trained model: "))
        sample, target = get_blood(pretrained)
        print("Probability: "+str((get_prediction(pretrained, sample)[0][0])))
        print("Actual: "+str(target))
