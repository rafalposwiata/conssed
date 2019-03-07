FROM nvcr.io/nvidia/tensorflow:18.06-py3
RUN apt update && apt install --yes software-properties-common python-software-properties \
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt update \
    && apt install --yes libc6 libc6-dev libc6-dev-i386 linux-libc-dev libsm6 libxext6 libgtk2.0-dev python3.6 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.6 \
    && apt install -y python3.6-dev python-dev \
    && python3.6 -m pip install emoji numpy tqdm colorama tensorflow==1.12.0 tensorflow-gpu==1.12.0 tensorflow_hub==0.2.0 keras ftfy nltk pandas lxml h5py preprocessor pycountry sklearn termcolor gensim ekphrasis text-unidecode fire twython bs4 hyperopt networkx==1.11

RUN mkdir /conssed
COPY . /conssed