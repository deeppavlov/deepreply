# REST API for DeepReply


Download models and embeddings
```sh
mkdir -p ~/deepreply_data
cd ~/deepreply_data
wget http://lnsigo.mipt.ru/export/models/insults.tar.gz
wget http://lnsigo.mipt.ru/export/models/paraphraser.tar.gz
wget http://lnsigo.mipt.ru/export/models/ner.tar.gz
wget http://lnsigo.mipt.ru/export/models/squad.tar.gz
wget http://lnsigo.mipt.ru/export/models/coreference.tar.gz
wget http://lnsigo.mipt.ru/export/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin
wget http://lnsigo.mipt.ru/export/embeddings/glove.840B.300d.txt
wget http://lnsigo.mipt.ru/export/embeddings/reddit_fasttext_model.bin
```
Build and run docker image
```sh
sudo docker build . -t deepreply/api

sudo docker run --rm -h api.local    \
           --name api                \
           -p 5000:80                \
           -v ~/deepreply_data:/data \
            deepreply/api:latest
```
