FROM bluelens/tensorflow:1.3.0-py3

#ENV WEB_CONCURRENCY=4
ENV OD_MODEL=/usr/src/app/model/frozen_inference_graph.pb
ENV OD_LABELS=/usr/src/app/model/label_map.pbtxt
ENV CLASSIFY_GRAPH=/usr/src/app/model/classify_image_graph_def.pb

RUN mkdir -p /usr/src/app
RUN mkdir -p /dataset/deepfashion

WORKDIR /usr/src/app

COPY . /usr/src/app

#RUN apt-get install ca-certificates libffi6 libstdc++ && \
#    apt-get install --virtual build-deps build-base libffi-dev && \
RUN pip install --no-cache-dir gunicorn /usr/src/app

EXPOSE 8080

CMD ["gunicorn", "-k", "gevent", "--timeout", "200", "-b", "0.0.0.0:8080", "bl_api_search:app"]
