import numpy as np
import time
from util import s3

import json
import tensorflow as tf
from PIL import Image
import os
import uuid
import redis
from os import listdir
from os.path import isfile, join
from util import label_map_util, s3
from stylelens_feature import feature_extract
import stylelens_search_vector
from stylelens_search_vector.rest import ApiException
from pprint import pprint


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

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

REDIS_SERVER = os.environ['REDIS_SERVER']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']

REDIS_KEY_IMAGE_HASH = 'bl:image:hash'
REDIS_KEY_IMAGE_LIST = 'bl:image:list'
rconn = redis.StrictRedis(REDIS_SERVER, port=6379, password=REDIS_PASSWORD)

class Search:
  def __init__(self):
    print('init')
    self.image_feature = feature_extract.ExtractFeature()
    self.vector_search_client = stylelens_search_vector.SearchApi()

  def search_imgage(self, image_file):
    if image_file and self.allowed_file(image_file.filename):
      im = Image.open(image_file.stream)

      file_type = image_file.filename.rsplit('.', 1)[1]
      if 'jpg' == file_type or 'JPG' == file_type or 'jpeg' == file_type or 'JPEG' == file_type:
        print('jpg')
      else:
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im, (0,0), im)
        bg.save('file.jpg', quality=95)
        im = bg
      im.show()
      size = 300, 300
      im.thumbnail(size, Image.ANTIALIAS)
      # im.show()
      file_name = str(uuid.uuid4()) + '.jpg'
      im.save(file_name)
      feature = self.extract_feature(file_name)
      print(feature.dtype)
      return self.query_feature(feature.tolist())

  def query_feature(self, vector):
    body = stylelens_search_vector.VectorSearchRequest() # VectorSearchRequest |
    body.vector = vector

    try:
      # Query to search vector
      api_response = self.vector_search_client.search_vector(body)
      pprint(api_response)
    except ApiException as e:
      print("Exception when calling SearchApi->search_vector: %s\n" % e)

    if api_response.data.vector is not None:
      res_vector = api_response.data.vector
      pprint(res_vector)

    response_images = []
    for idx in res_vector:
      print(idx)
      image_info = self.get_image_info(idx)
      response_images.append(image_info)

    return response_images

  def allowed_file(self, filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

  def extract_feature(self, file):
    feature = self.image_feature.extract_feature(file)
    # print(feature)
    return feature

  def get_image_info(self, index):
    image_id = rconn.lindex(REDIS_KEY_IMAGE_LIST, index-1)
    if type(image_id) is bytes:
      image_id = image_id.decode('utf-8')

    data = rconn.hget(REDIS_KEY_IMAGE_HASH, image_id)
    print(data)
    image_info = json.loads(data.decode('utf-8'))
    # print(image_info)
    return image_info
