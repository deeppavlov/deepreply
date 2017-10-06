import os
import json
import requests
import numpy as np

from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent


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

    # Generate params for EnsembleParaphraserAgent
    def _make_agent_params(self):
        embeddings_dir = self.config['embeddings_dir']
        embedding_file = self.config['kpis'][self.kpi_name]['settings_agent']['fasttext_model']
        model_files = self.opt['model_files']
        agent_params = {
            'fasttext_model': os.path.join(embeddings_dir, embedding_file),
            'model_files': model_files,
            'datatype': 'test'
        }
        return agent_params

    # Initiate agent
    def init_agent(self):
        self.agent = EnsembleParaphraserAgent(self._make_agent_params())

    # Update Tester config with or without [re]initiating agent
    def update_config(self, config, init_agent=False):
        self.config = config
        if init_agent:
            self.init_agent()

    # Update tasks number
    def set_numtasks(self, numtasks):
        self.numtasks = numtasks

    # Get kpi2 tasks via REST
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
                'text': 'Dummy title\n%s\n%s' % (task['phrase1'], task['phrase2']),
            })
        return observations

    # Process observations via algorithm
    def _get_predictions(self, observations):
        predictions = self.agent.batch_act(observations)
        return predictions

    # Generate answers data
    def _make_answers(self, session_id, observations, predictions):
        answers = {}
        answers['sessionId'] = session_id
        answers['answers'] = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers['answers'][obs['id']] = (lambda s: np.float64((1 if s == 0.5 else round(s))))(pred['score'][0])
        return answers

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

        answers = self._make_answers(session_id, observations, predictions)
        self.answers = answers

        score = self._get_score(answers)
        self.score = score