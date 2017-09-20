import os
import json
import requests
import numpy as np

from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent


class tester():

    def __init__(self, config, opt):
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

    # Get kpi2 tasks via REST
    def get_tasks(self):
        get_url = self.config['kpis'][self.kpi_name]['rest_url']
        test_tasks_number = self.config['kpis'][self.kpi_name]['test_tasks_number']
        get_params = {'stage': 'test', 'quantity': test_tasks_number}
        get_response = requests.get(get_url, params=get_params)
        tasks = json.loads(get_response.text)
        return tasks

    # Prepare observations set
    def make_observations(self, tasks):
        observations = []
        for task in tasks['qas']:
            observations.append({
                'id': task['id'],
                'text': 'Dummy title\n%s\n%s' % (task['phrase1'], task['phrase2']),
            })
        return observations

    # Generate params for EnsembleParaphraserAgent
    def make_agent_params(self):
        embeddings_dir = self.config['embeddings_dir']
        embedding_name = self.config['kpis'][self.kpi_name]['use_embedding']
        embeddings_file = self.config['embeddings'][embedding_name]['files']
        models_files = self.opt['models_files']
        agent_params = {
            'fasttext_model': os.path.join(embeddings_dir, embeddings_file),
            'model_files': models_files,
            'datatype': 'test'
        }
        return agent_params

    # Process observations via algorithm
    def get_predictions(self, opt, observations):
        agent = EnsembleParaphraserAgent(opt)
        predictions = agent.batch_act(observations)
        return predictions

    # Generate answers data
    def make_answers(self, session_id, observations, predictions):
        answers = {}
        answers['sessionId'] = session_id
        answers['answers'] = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers['answers'][obs['id']] = (lambda s: np.float64((1 if s == 0.5 else round(s))))(pred['score'][0])
        return answers

    # Post answers data and get score
    def get_score(self, answers):
        post_headers = {'Accept': '*/*'}
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['rest_url'], \
                                      json=answers, \
                                      headers=post_headers)
        return rest_response.text

    # Run full cycle of testing session and store data for each step
    def run_test(self):
        tasks = self.get_tasks()
        session_id = tasks['id']
        numtasks = tasks['total']
        self.tasks = tasks
        self.session_id = session_id
        self.numtasks = numtasks

        observations = self.make_observations(tasks)
        self.observations = observations

        agent_params = self.make_agent_params()
        self.agent_params = agent_params

        predictions = self.get_predictions(agent_params, observations)
        self.predictions = predictions

        answers = self.make_answers(session_id, observations, predictions)
        self.answers = answers

        score = self.get_score(answers)
        self.score = score