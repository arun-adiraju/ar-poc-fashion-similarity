#!/usr/bin/python
from labelImage import label
from similaritySearch import find_similar, get_similar_images_using_session


def get_label(file_name):
    label_name = label(file_name)
    similar_images = find_similar(file_name, label_name)
    print(similar_images)
    print("label name in service" + label_name)
    return similar_images, label_name


def get_image_label(file_name):
    label_name = label(file_name)
    return label_name


def get_similar_images(test_file_name):
    label_name = label(test_file_name)
    similar_images = get_similar_images_using_session(test_file_name, label_name)
    return similar_images, label_name

    # def get_all_feature_vectors():
    #     return get_label_features_mapping()
