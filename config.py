import os

DATA_DIR = './build/'
MODELS_DIR = './build/models/'
EMBEDDINGS_DIR = './build/'

REPO_URL = os.getenv('MODELS_URL')
REPO_MODEL_URL = 'paraphraser/paraphraser_models_final.tar.gz'

MODEL_NAME = 'maxpool_match'
ENSEMBLE_MODELS_NUM = 5
#FASTTEXT_MODEL_FILENAME = ''

EMBEDDINGS_FILENAME = 'ft_0.8.3_nltk_yalen_sg_300.bin'

REST_URL = 'http://api.aibotbench.com/kpi2/qas'
TEST_TASKS_NUMBER = 5