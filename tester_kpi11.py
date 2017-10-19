import os
import json
import requests
import re
import copy

import build_utils as bu
from parlai.core.agents import create_agent


class Tester:

    def __init__(self, config, opt):
        self.agent = None
        self.config = config
        self.opt = opt
        self.kpi_name = config['kpi_name']
        self.session_id = None
        self.numtasks = None
        self.tasks = None
        self.observations = None
        self.agent_params = None
        self.predictions = None
        self.answers = None
        self.score = None

    # Initiate agent
    def init_agent(self):
        params = ['-t', 'deeppavlov.tasks.coreference_scorer_model.agents:CoreferenceTeacher',
                    '-m', 'deeppavlov.agents.coreference_scorer_model.agents:CoreferenceAgent',
                    '--display-examples', 'False',
                    '--num-epochs', '20',
                    '--log-every-n-secs', '-1',
                    '--log-every-n-epochs', '1',
                    '--validation-every-n-epochs', '1',
                    '--chosen-metrics', 'f1',
                    '--validation-patience', '20',
                    '--datatype', 'test']
        opt = bu.arg_parse(params)
        embeddings_dir = self.config['embeddings_dir']
        embedding_file = self.config['kpis'][self.kpi_name]['settings_agent']['embedding_file']
        model_files = self.opt['model_files']
        opt['model_file'] = os.path.dirname(model_files[0])
        opt['pretrained_model'] = os.path.dirname(model_files[0])
        opt['embeddings_path'] = os.path.join(embeddings_dir, embedding_file)
        print(opt)
        self.agent = create_agent(opt)

    # Update Tester config with or without [re]initiating agent
    def update_config(self, config, init_agent=False):
        self.config = config
        if init_agent:
            self.init_agent()

    # Update tasks number
    def set_numtasks(self, numtasks):
        self.numtasks = numtasks

    # Get kpi11 tasks via REST
    def _get_tasks(self):
        get_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url']
        if self.numtasks in [None, 0]:
            test_tasks_number = self.config['kpis'][self.kpi_name]['settings_kpi']['test_tasks_number']
        else:
            test_tasks_number = self.numtasks
        get_params = {'stage': 'test', 'quantity': test_tasks_number}
        get_response = requests.get(get_url, params=get_params)
        tasks = json.loads(get_response.text)
        return tasks

    # Prepare observations set
    def _make_observations(self, tasks):
        id = []
        observation = {'conll': [], 'valid_conll': [], 'id': ''}
        for task in tasks['qas']:
            conll_str = str(task['question'])

            # Preprocess conll
            doc_num = str(re.search(r'#begin document [(].+[)];\n([0-9]+)', conll_str).group(1))
            conll_str = re.sub(r'(?P<subst>#begin document [(].+[)];)',
                               '#begin document(%s); part 0' % doc_num,
                               conll_str)
            match = re.search(r'\n\n#end document', conll_str)
            if match is None:
                conll_str = re.sub(r'\n#end document', r'\n\n#end document', conll_str)

            # Prepare task
            id.append(task['id'])
            observation['valid_conll'].append(conll_str.split('\n'))

        observations = {'id': id, 'observation': observation}
        return observations

    def _extract_coref(self, conll):
        coref_str = ''
        lines = conll.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith("#begin"):
                coref_str += ' '
            elif lines[i].startswith("#end document"):
                coref_str += ' '
            else:
                row = lines[i].split('\t')
                if len(row) == 1:
                    coref_str += ' '
                else:
                    coref_str += row[-1] + ' '
        return coref_str

    # Process observations via algorithm
    def _get_predictions(self, observations):
        self.agent.observe(observations['observation'])
        predictions = self.agent.act()
        return predictions

    # Generate answers data
    def _make_answers(self, observations, predictions):
        id_predict = {}
        observe_predict = list(zip(observations['id'], predictions['valid_conll']))
        for obs, pred in observe_predict:
            id_predict[obs] = self._extract_coref(''.join(pred))
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = id_predict
        # Reduce POST request size
        tasks['qas'] = []
        return tasks

    # Post answers data and get score
    def _get_score(self, answers):
        post_headers = {'Accept': '*/*'}
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url'],
                                      json=answers,
                                      headers=post_headers)
        return rest_response.text

    # Run full cycle of testing session and store data for each step
    def run_test(self, init_agent=True):
        if init_agent:
            self.init_agent()

        tasks = self._get_tasks()
        session_id = tasks['id']
        numtasks = tasks['total']
        self.tasks = tasks
        self.session_id = session_id
        self.numtasks = numtasks

        observations = self._make_observations(tasks)
        self.observations = observations

        predictions = self._get_predictions(observations)
        self.predictions = predictions

        answers = self._make_answers(observations, predictions)
        self.answers = answers

        score = self._get_score(answers)
        self.score = score
