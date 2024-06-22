#!/bin/bash
DOCKERHUB_USERNAME=mengyibai
IMAGE_NAME=shallenge
TAG=latest

docker run --rm -w /workspace -v .:/workspace --user $(id -u):$(id -g) nvidia/cuda:12.4.0-devel-ubuntu22.04 make

DOCKERFILE=$(mktemp)
cat <<EOF > $DOCKERFILE
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY build/shallenge /workspace/shallenge
ENTRYPOINT [ "/workspace/shallenge" ]
EOF

docker build -t ${IMAGE_NAME}:${TAG} -f $DOCKERFILE .
docker tag ${IMAGE_NAME}:${TAG} ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:$TAG
docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}