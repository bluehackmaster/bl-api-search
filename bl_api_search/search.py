import numpy as np
import time
from util import s3

import json
import tensorflow as tf
from PIL import Image
import os
import uuid
from os import listdir
from os.path import isfile, join
from util import label_map_util, s3
from stylelens_feature import feature_extract

IMG_NUM = 1408
QUERY_IMG = 22
CANDIDATES = 5

NUM_CLASSES = 89

STR_BUCKET = "bucket"
STR_STORAGE = "storage"
STR_CLASS_CODE = "class_code"
STR_NAME = "name"
STR_FORMAT = "format"

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

OD_MODEL = os.environ['OD_MODEL']
OD_LABELS = os.environ['OD_LABELS']

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

class Search:
  def __init__(self):
    print('init')
    label_map = label_map_util.load_labelmap(OD_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    self.__category_index = label_map_util.create_category_index(categories)
    self.__detection_graph = tf.Graph()
    with self.__detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(OD_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

      self.__sess = tf.Session(graph=self.__detection_graph)

    self.image_feature = feature_extract.ExtractFeature()

  def search_imgage(self, image_file):
    if image_file and self.allowed_file(image_file.filename):
      im = Image.open(image_file.stream)
      im.show()
      size = 300, 300
      im.thumbnail(size, Image.ANTIALIAS)
      im.show()
      file_name = str(uuid.uuid4()) + '.jpg'
      im.save(file_name)
      self.extract_feature(file_name)


  def allowed_file(self, filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

  def extract_feature(self, file):
    feature = self.image_feature.extract_feature(file)
    print(feature)

