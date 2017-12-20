import os
import sys

from PIL import Image
from getFeatures import extract_features
from numpy import genfromtxt
from scipy.spatial import distance

uploadedImageFolder = "static/uploadedImages/"
bottleneck_dir = '/Users/1020621/ml/deepLearning/self/code/fashionSimilarity/tensorflow-for-poets-2/tf_files/bottlenecks'
image_dir = '/Users/1020621/ml/deepLearning/self/code/fashionSimilarity/tensorflow-for-poets-2/tf_files/inputFiles/'
_GLOBAL_FEATURE_MAPPINGS = {}


# class to store each image data
class ImageData:
    def __init__(self, features=[] * 10, name='', distance=sys.maxsize):
        self.name = name
        self.features = features
        self.distance = distance

    def get_features(self):
        return self.features

    def set_features(self, features):
        self.features = features

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_distance(self):
        return self.distance

    def set_distance(self, distance):
        self.distance = distance

    def __str__(self):
        return "(" + str(self.name) + ", " + str(self.features) + ", " + str(self.distance)


# read bottleneck data from each file
def read_bottleneck_data(dir, file_name):
    test_image_object = ImageData()
    my_data = genfromtxt(dir + file_name, delimiter=',')
    test_image_object.set_features(my_data)
    test_image_object.set_name(file_name)
    return test_image_object


# get image path
def get_image_path(image_data):
    tokens = image_data.get_name().split("/")[-1].split(".")
    tf_extension = tokens[1].split("_")
    return tokens[0] + "." + tf_extension[0]


# display iamge given ImageData Object
def display_image(image_data):
    tokens = image_data.get_name().split("/")[-1].split(".")
    tf_extension = tokens[1].split("_")
    image = Image.open(image_dir + tokens[0] + "." + tf_extension[0])
    image.show()


def get_test_file_features(test_file_name):
    return extract_features(uploadedImageFolder + test_file_name)


def find_similar(test_file_name, label):
    test_file_features = get_test_file_features(test_file_name)

    all_feature_vectors = []
    for file in os.listdir(bottleneck_dir + "/" + label + "/"):
        if file.endswith(".txt"):
            each_image_data = read_bottleneck_data(bottleneck_dir + "/" + label + "/", file)
            dst = distance.euclidean(each_image_data.get_features(), test_file_features)
            each_image_data.set_distance(dst)
            each_image_data.set_name(image_dir + label + "/" + file)
            all_feature_vectors.append(each_image_data)

    all_feature_vectors.sort(key=lambda x: x.distance)

    similar_image_paths = []
    top_5_similar_images = all_feature_vectors[:5]
    for each_similar_image_data in top_5_similar_images:
        similar_image_paths.append(get_image_path(each_similar_image_data))

    return similar_image_paths


def get_label_features_mapping1():
    lables = next(os.walk(bottleneck_dir))[1]
    label_features_mapping = {}

    for each_label in lables:
        all_feature_vectors = []
        for file in os.listdir(bottleneck_dir + "/" + each_label + "/"):
            if file.endswith(".txt"):
                each_image_data = read_bottleneck_data(bottleneck_dir + "/" + each_label + "/", file)
                '''print(each_image_data) '''
                each_image_data.set_name(image_dir + each_label + "/" + file)
                each_image_data.set_features(each_image_data.features)
                all_feature_vectors.append(each_image_data)

        label_features_mapping[each_label.lower()] = all_feature_vectors
        print('done loading features for ' + each_label)

    # print(label_features_mapping)
    global _GLOBAL_FEATURE_MAPPINGS
    _GLOBAL_FEATURE_MAPPINGS = label_features_mapping
    print(_GLOBAL_FEATURE_MAPPINGS)
    return label_features_mapping


def get_similar_images_using_session(test_file_name, label):
    test_file_features = get_test_file_features(test_file_name)
    # print("Ankit: " + _GLOBAL_FEATURE_MAPPINGS)
    image_data_list = _GLOBAL_FEATURE_MAPPINGS.get(label)
    for each_image_data in image_data_list:
        dst = distance.euclidean(each_image_data.get_features(), test_file_features)
        each_image_data.set_distance(dst)


    print(image_data_list)
    image_data_list.sort(key=lambda x: x.distance)
    similar_image_paths = []
    top_5_similar_images = image_data_list[:5]

    for each_similar_image_data in top_5_similar_images:
        image_paths = get_image_path(each_similar_image_data)
        print(image_paths)
        similar_image_paths.append(image_paths)

    return similar_image_paths


def test():
    print("test method ")
