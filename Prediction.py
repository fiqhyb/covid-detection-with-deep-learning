import numpy as np
import tensorflow as tf
import requests


def get_blood():
    r = requests.get('http://ec2-3-25-181-52.ap-southeast-2.compute.amazonaws.com/api/v1/blood-app/openapi.json')
    values = list(r.json()['components']['schemas']['Data']['example']['data'].values())
    return values


def model_loader(num):
    switcher = {
        1: "San_Raphael.h5",
        2: "Albert_Einstein.h5",
        3: "Pasar_Minggu.h5"
    }
    return switcher.get(num, "no model")


def get_prediction(model_file, blood_input):
    model = tf.keras.models.load_model(model_loader(model_file),
                                       custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    arr = np.array(blood_input).reshape(1, len(blood_input))
    return model.predict(arr)


if __name__ == '__main__':
    print((get_prediction(1, get_blood())[0][0]))
