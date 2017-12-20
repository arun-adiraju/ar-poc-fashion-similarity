import csv
import sys
import os

from numpy import genfromtxt
from scipy.spatial import distance
from PIL import Image


test_file_name = 'n02780704_7_0.jpg_inception_v3.txt'
bottleneck_dir = '/Users/1020621/ml/deepLearning/self/code/fashionSimilarity/tensorflow-for-poets-2/tf_files/bottlenecks/LongGown/'
image_dir = '/Users/1020621/ml/deepLearning/self/code/fashionSimilarity/tensorflow-for-poets-2/tf_files/inputFiles/LongGown/'

# class to store each image data
class ImageData:

	def __init__(self, features=[]*10, name='', distance=sys.maxsize):
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

# display iamge given ImageData Object
def display_image(image_data):
	tokens = image_data.get_name().split("/")[-1].split(".")
	tf_extension = tokens[1].split("_")
	image = Image.open(image_dir + tokens[0] + "." + tf_extension[0])
	image.show()

all_feature_vectors = []
test_file_features = read_bottleneck_data(bottleneck_dir, test_file_name).get_features()

for file in os.listdir(bottleneck_dir):
    if file.endswith(".txt"):
    	each_image_data = read_bottleneck_data(bottleneck_dir, file)
    	dst = distance.euclidean(each_image_data.get_features(), test_file_features)
    	each_image_data.set_distance(dst)
    	each_image_data.set_name(image_dir + file)
    	all_feature_vectors.append(each_image_data)
    

all_feature_vectors.sort(key=lambda x: x.distance)

for each_similar_image_data in all_feature_vectors[:5]:
	print(str(each_similar_image_data))
	display_image(each_similar_image_data)
	




