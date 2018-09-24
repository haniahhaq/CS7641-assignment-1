# Georgia Tech OMSCS - CS7641 - Machine Learning Assignment 1 Code

### How to run this code



I recommend using Docker. The image used to run this code is available on Dockerhub: [https://hub.docker.com/r/wcsmith/ml-notebook/](https://hub.docker.com/r/wcsmith/ml-notebook/)

Otherwise, you'll need a Python 3.x environment with Scikit-learn, Keras, and a Jupyter notebook environment.

All plots were generated using the Jupyter notebook files starting with nb_

All data is stored in this repository (yes, including all of MNIST). It should work right out of the box.

Step-by-step:

1. Clone this repository: [https://github.com/wesley-smith/ml-assignment1.git](https://github.com/wesley-smith/ml-assignment1.git)
2. cd to that directory
3. Start the container `docker run --rm -d -p 8888:8888 -v "$PWD":/home/jovyan/ --name ml-a1 wcsmith/ml-notebook:latest start-notebook.sh --NotebookApp.token='token'`
4. Connect your browser to http://127.0.0.1:8888/?token=token
