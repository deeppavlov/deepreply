FROM python:3.6-stretch
WORKDIR /app
VOLUME /data
EXPOSE 5000
RUN git clone -b no_gpu https://github.com/deepmipt/deeppavlov && \
    cd deeppavlov && \
    pip install Cython==0.26.0 && \
    pip install -r requirements.txt && \
    python setup.py develop && \
    cd .. && \
    git clone -b api2 https://github.com/deepmipt/deepreply && \
    cd deepreply && \
    pip install -r requirements.txt
CMD cd deepreply && python api.py
