import os
import json
import requests

from parlai.core.params import ParlaiParser
from deeppavlov.agents.ner.ner import NERAgent


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

    # Get kpi1 tasks via REST
    def get_tasks(self):
        get_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url']
        test_tasks_number = self.config['kpis'][self.kpi_name]['settings_kpi']['test_tasks_number']
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
                'text': task['question']
            })
        return observations

    # Generate params for EnsembleParaphraserAgent
    def make_agent_params(self):
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
        agent_params['model_file'] = os.path.dirname(model_files[0])
        agent_params['dict_file'] = os.path.join(os.path.dirname(model_files[0]), agent_settings['dict_files_names'])
        agent_params['display_examples'] = bool(agent_settings['display_examples'])
        return agent_params

    # Process observations via algorithm
    def get_predictions(self, opt, observations):
        agent = NERAgent(opt)
        predictions = agent.batch_act(observations)
        return predictions

    # Generate answers data
    def make_answers(self, session_id, observations, predictions):
        answers = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers[obs['id']] = pred['text']
            #import re
            #answers[obs['id']] = re.sub('__\w+__', '0', pred['text'])
        #tasks = self.tasks
        #tasks['answers'] = answers
        #return tasks
        answ = {}
        answ['answers'] = answers
        return answ

    # Post answers data and get score
    def get_score(self, answers):
        post_headers = {'Accept': '*/*'}
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url'], \
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
        print(tasks)
        print(session_id)
        print(numtasks)

        observations = self.make_observations(tasks)
        self.observations = observations
        print(observations)

        agent_params = self.make_agent_params()
        self.agent_params = agent_params
        print(agent_params)

        predictions = self.get_predictions(agent_params, observations)
        self.predictions = predictions
        print(predictions)

        answers = self.make_answers(session_id, observations, predictions)
        self.answers = answers
        print(answers)
        print(json.dumps(answers))

        score = self.get_score(answers)
        self.score = score