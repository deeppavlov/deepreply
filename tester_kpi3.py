import os
import json
import requests

from parlai.core.params import ParlaiParser
from deeppavlov.agents.ner.ner import NERAgent


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

    # Generate params for NERAgent
    def _make_agent_params(self):
        parser = ParlaiParser(True, True)
        NERAgent.add_cmdline_args(parser)
        args = ['-t deeppavlov.tasks.ner.agents',
                '-m deeppavlov.agents.ner.ner:NERAgent']
        agent_params = parser.parse_args(args=args)
        agent_settings = self.config['kpis'][self.kpi_name]['settings_agent']
        for key, value in agent_settings.items():
            if key not in ['model_files_names', 'dict_files_names', 'display_examples']:
                agent_params[key] = value
        model_files = self.opt['model_files']
        # model_file and pretrained_model params should always point to the same dir with pretrained model
        agent_params['model_file'] = os.path.dirname(model_files[0])
        agent_params['pretrained_model'] = os.path.dirname(model_files[0])
        agent_params['dict_file'] = os.path.join(os.path.dirname(model_files[0]), agent_settings['dict_files_names'])
        agent_params['display_examples'] = bool(agent_settings['display_examples'])
        return agent_params

    # Initiate agent
    def init_agent(self):
        self.agent = NERAgent(self._make_agent_params())

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
                'text': task['question']
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
        tasks = self.tasks
        tasks['answers'] = answers
        return tasks
        # answ = {}
        # answ['answers'] = answers
        # return answ

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

        agent_params = self._make_agent_params()
        self.agent_params = agent_params
        predictions = self._get_predictions(observations)
        self.predictions = predictions

        answers = self._make_answers(observations, predictions)
        self.answers = answers

        score = self._get_score(answers)
        self.score = score
