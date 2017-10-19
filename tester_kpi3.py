import os
import json
import requests
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
        params = ['-t', 'deeppavlov.tasks.ner.agents',
            '-m', 'deeppavlov.agents.ner.ner:NERAgent',
            '-dt', 'test',
            '--batchsize', '2',
            '--display-examples', 'False',
            '--validation-every-n-epochs', '5',
            '--log-every-n-epochs', '1',
            '--log-every-n-secs', '-1',
            '--chosen-metrics', 'f1']
        dict_file = self.config['kpis'][self.kpi_name]['settings_agent']['dict_files_names']
        model_files = self.opt['model_files']
        opt = bu.arg_parse(params)
        opt['model_file'] = os.path.dirname(model_files[0])
        opt['pretrained_model'] = os.path.dirname(model_files[0])
        opt['dict_file'] = os.path.join(os.path.dirname(model_files[0]), dict_file)
        self.agent = create_agent(opt)


    # Update Tester config with or without [re]initiating agent
    def update_config(self, config, init_agent=False):
        self.config = config
        if init_agent:
            self.init_agent()

    # Update tasks number
    def set_numtasks(self, numtasks):
        self.numtasks = numtasks

    # Get kpi3 tasks via REST
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
        observations = []
        for task in tasks['qas']:
            observations.append({
                'id': task['id'],
                # Preprocess task
                'text': task['question'].split('\t')[0]
            })
        return observations

    # Process observations via algorithm
    def _get_predictions(self, observations):
        raw_predicts = self.agent.batch_act(observations)
        predictions = [{'id': pred['id'], 'text': pred['text'].replace('__NULL__', '').strip()}
                       for pred in raw_predicts]
        return predictions

    # Generate answers data
    def _make_answers(self, observations, predictions):
        answers = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers[obs['id']] = pred['text']
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = answers
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
