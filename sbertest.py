import os
import shutil
import json
import urllib.request
import tarfile


def get_models_files(config, models_repo_url):
    kpi_name = config['kpi_name']
    models = [config['models'][model] for model in config['kpis'][kpi_name]['use_models']]
    models_dir = config['models_dir']
    update_models = config['update_models']
    models_files = []
    for model in models:
        # Prepare paths for model downloading and storing
        model_url = urljoin(models_repo_url, model['repo_url'])
        store_path = urljoin(models_dir, model['repo_url'])
        model_dir = urljoin(os.path.dirname(store_path), \
                            os.path.basename(store_path).split('.', maxsplit=-1)[0]) + '/'

        # Prepare models files list
        for model_file in model['files']:
            models_files.append(model_dir + model_file)
        print('Models files list:')
        print('    %s' % (str(models_files)))

        if update_models:
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

    return models_files


def urljoin(*args):
    url = '/'.join(map(lambda x: str(x).rstrip('/').lstrip('/'), args))
    return '/' + url if args[0][0] == '/' else url


def main():
    # Initialise optional params dict for further usage
    opt = {}

    # Initialise environment variables
    print('Reading environment variables...')
    opt['models_repo_url'] = os.getenv('MODELS_URL')
    opt['embeddings_repo_url'] = os.getenv('EMBEDDINGS_URL')
    opt['datasets_repo_url'] = os.getenv('DATASETS_URL')

    # Read config.json
    print('Reading config.json...')
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Make models data dir
    data_dir = config['data_dir']
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    # Get models files list [and update models files]
    print('Getting models files...')
    opt['models_files'] = get_models_files(config, opt['models_repo_url'])

    # Execute test
    print('Executing %s test...' % config['kpi_name'])
    tester_module = __import__('tester_' + config['kpi_name'])
    tester_class = getattr(tester_module, 'tester')
    tester = tester_class(config, opt)
    tester.run_test()
    print('%s test finished, tasks number: %s, SCORE: %s' % (config['kpi_name'], str(tester.numtasks), str(tester.score)))

if __name__ == '__main__':
    main()


