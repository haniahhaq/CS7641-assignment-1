#!/bin/bash
docker run --rm -d -p 8888:8888 -v "$PWD":/home/jovyan/ --name ml-a1 wcsmith/ml-notebook:latest start-notebook.sh --NotebookApp.token='token' 
