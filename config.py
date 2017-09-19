import os

DATA_DIR = './build/'
MODELS_DIR = './build/models/'
COMMON_MODELS_DIR = './build/models/common/'

REPO_URL = os.getenv('MODELS_URL')
REPO_MODEL_URL = 'paraphraser/paraphraser_models_final.tar.gz'

MODEL_NAME = ''
ENSEMBLE_MODELS_NUM = 5
FASTTEXT_MODEL_FILENAME = ''