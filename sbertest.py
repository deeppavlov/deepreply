import os
import shutil
import json
import requests
import urllib.request
import tarfile
import numpy as np

from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent

import config


#def run_test(url, tasks_number):
def run_test(param_dict):
    #print('Number of tasks: %i' % tasks_number)
    print('Number of tasks: %i' % param_dict['test_tasks_number'])

    # Settings
    #data_dir = './build/'
    rep_models_url = os.environ.get('MODELS_URL')

    # Get tasks
    print('Getting tasks...')
    #get_params = {'stage': 'test', 'quantity': tasks_number}
    #get_response = requests.get(url, params=get_params)
    get_params = {'stage': 'test', 'quantity': param_dict['test_tasks_number']}
    get_response = requests.get(param_dict['rest_url'], params=get_params)
    tasks = json.loads(get_response.text)
    session_id = tasks['id']

    # Prepare observations set
    print('Preparing observation set...')
    observations = []
    for task in tasks['qas']:
        # Observations
        observations.append({
            'id': task['id'],
            'text': 'Dummy title\n%s\n%s' % (task['phrase1'], task['phrase2']),
        })

    # Process observations via algorithm
    print('Processing observations via algorithm...')
    opt = {
        'fasttext_model': os.path.join(param_dict['embeddings_dir'], 'ft_0.8.3_nltk_yalen_sg_300.bin'),
        #paraphraser/paraphraser_models_final
        #'model_files': [os.path.join(param_dict['models_dir'], \
        'model_files': [os.path.join((param_dict['models_dir'] + 'paraphraser/paraphraser_models_final/'), \
                                  param_dict['model_name'] + '_' + str(i)) \
                        for i in range(param_dict['test_tasks_number'])],
        #'fasttext_model': os.path.join(data_dir, 'ft_0.8.3_nltk_yalen_sg_300.bin'),
        #'model_files': [os.path.join(data_dir, 'paraphraser_%i' % i) for i in range(5)],
        'datatype': 'test'
    }
    agent = EnsembleParaphraserAgent(opt)
    predictions = agent.batch_act(observations)

    # Prepare answers data
    print('Preparing answers data...')
    post_dict = {}
    post_dict['sessionId'] = session_id
    post_dict['answers'] = {}
    obs_pred = list(zip(observations, predictions))
    for obs, pred in obs_pred:
        post_dict['answers'][obs['id']] = (lambda s: np.float64((1 if s == 0.5 else round(s))))(pred['score'][0])

    # MAINTENANCE!
    import pickle
    dump_get = './dump_get.pickle'
    dump_obs = './dump_obs.pickle'
    dump_pred = './dump_pred.pickle'
    dump_post = './dump_post.pickle'
    with open(dump_get, 'wb') as f:
        pickle.dump(get_response, f)
    with open(dump_obs, 'wb') as f:
        pickle.dump(observations, f)
    with open(dump_pred, 'wb') as f:
        pickle.dump(predictions, f)
    with open(dump_post, 'wb') as f:
        pickle.dump(post_dict, f)
    #print('get_session_resp:')
    #print(tasks.text)
    #print('observations:')
    #print(observations)
    #print('predictions:')
    #print(predictions)

    # Post answers data
    print('Posting answers data...')
    post_headers = {'Accept': '*/*'}
    rest_response = requests.post(param_dict['rest_url'], json=post_dict, headers=post_headers)
    print('Matching result: %s' % rest_response.text)


def update_models(param_dict):
    # Prepare paths for model downloading and storing
    model_url = urljoin(param_dict['repo_url'], param_dict['repo_model_url'])
    store_path = urljoin(param_dict['models_dir'], param_dict['repo_model_url'])
    model_dir = urljoin(os.path.dirname(store_path), \
                        os.path.basename(store_path).split('.', maxsplit=-1)[0]) + '/'

    # Download model
    print('Downloading model to ' + store_path + ' ...')
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    if os.path.isdir(model_dir) and model_dir != '/':
        shutil.rmtree(model_dir, ignore_errors=True)
    req = urllib.request.urlopen(model_url)
    with open(store_path, 'w+b') as f:
        f.write(req.read())
        f.close()
    print('Downloading model finished')

    # Extract model
    print('Extracting model to ' + model_dir + ' ...')
    tar = tarfile.open(store_path, 'r:gz')
    tar.extractall(path=os.path.dirname(store_path))
    tar.close()
    print('Extracting model finished')


def urljoin(*args):
    url = '/'.join(map(lambda x: str(x).rstrip('/').lstrip('/'), args))
    return '/' + url if args[0][0] == '/' else url


def main():
    # Initialise params
    #data_dir = config.DATA_DIR
    embeddings_dir = config.EMBEDDINGS_DIR

    # Initialise update model params
    update_model_param_dict = {}
    update_model_param_dict['models_dir'] = config.MODELS_DIR
    update_model_param_dict['repo_url'] = config.REPO_URL
    update_model_param_dict['repo_model_url'] = config.REPO_MODEL_URL
    update_models(update_model_param_dict)

    # Initialise test params
    test_param_dict = {}
    test_param_dict['embeddings_dir'] = config.EMBEDDINGS_DIR
    test_param_dict['embeddings_filename'] = config.EMBEDDINGS_FILENAME
    test_param_dict['models_dir'] = config.MODELS_DIR
    test_param_dict['model_name'] = config.MODEL_NAME
    test_param_dict['ensemble_models_num'] = config.ENSEMBLE_MODELS_NUM
    test_param_dict['rest_url'] = config.REST_URL
    test_param_dict['test_tasks_number'] = config.TEST_TASKS_NUMBER

    # Execute test
    #run_test('http://api.aibotbench.com/kpi2/qas', 5)
    run_test(test_param_dict)

if __name__ == '__main__':
    main()


