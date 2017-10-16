FROM python:3-alpine

ENV WEB_CONCURRENCY=4

ADD . /bl_api_search

RUN apk add -U ca-certificates libffi libstdc++ && \
    apk add --virtual build-deps build-base libffi-dev && \
    # Pip
    pip install --no-cache-dir gunicorn /bl_api_search && \
    # Cleaning up
    apk del build-deps && \
    rm -rf /var/cache/apk/*

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "bl_api_search:app"]
