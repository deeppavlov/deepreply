# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import requests
import copy

import build_utils as bu
from parlai.core.agents import create_agent

from multiprocessing import Process, Queue

class Tester(Process):
    """NER agent with KPI3 testing methods and data

    Properties:
        agent: KPI's model agent object
        config: dict object initialised with config.json and modified with run script
        opt: dict object with optional agent and KPI testing parameters
        kpi_name: string with KPI name
        session_id: string with testing session ID received from the testing system
        numtasks: integer with tasks number
        tasks: dict object initialised with tasks JSON received from the testing system
        observations: list object with observation set, prepared for the agent
        predictions: list object with results of agent inference on observations set
        answers: list object prepared for dumping into JSON payload of POST request according testing the system API
        score: string with result of agent predictions scoring by testing system
        response_code: string with the code of the testing system POST request response

    Public methods:
        init_agent(self): initiates model agent
        update_config(self, config, init_agent=False): updates Tester instance config
        set_numtasks(self, numtasks): updates Tester instance tasks number
        run_test(self, init_agent=True): evokes full cycle of KPI testing sequence with current config and tasks number
    """

    def __init__(self, config, opt, input_queue, output_queue):
        """Tester class constructor

        :param config: dict object initialised with config.json and modified with run script
        :type config: dict
        :param opt: dict object with optional agent and KPI testing parameters
        :type opt: dict
        """
        super(Tester, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.agent = None
        self.config = copy.deepcopy(config)
        self.opt = copy.deepcopy(opt)
        self.kpi_name = config['kpi_name']
        self.session_id = None
        self.numtasks = None
        self.tasks = None
        self.observations = None
        self.agent_params = None
        self.predictions = None
        self.answers = None
        self.score = None
        self.response_code = None

    def init_agent(self):
        """Initiate model agent
        """
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


    def update_config(self, config, init_agent=False):
        """Update Tester instance configuration dict

        Args:
            :param config: dict object initialised with config.json and modified with run script
            :type config: dict
            :param init_agent: integer flag (0, 1), turns off/on agent [re]initialising
            :type init_agent: int
        """
        self.config = config
        if init_agent:
            self.init_agent()

    def set_numtasks(self, numtasks):
        """Update Tester instance number of tasks, requested during the next testing session

        Args:
            :param numtasks: integer with tasks number
            :type numtasks: int
        Method is used when need in tasks number different to provided in config arises.
        In order to reset tasks number to config value, evoke this method with numtasks==0
        """
        self.numtasks = numtasks

    def _get_tasks(self):
        """Send GET request to testing system and get tasks set

        Returns:
            :return: dict object initialised with tasks JSON received from the testing system
            :rtype: dict
        """
        get_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url']
        if self.numtasks in [None, 0]:
            test_tasks_number = self.config['kpis'][self.kpi_name]['settings_kpi']['test_tasks_number']
        else:
            test_tasks_number = self.numtasks
        get_params = {'stage': 'test', 'quantity': test_tasks_number}
        get_response = requests.get(get_url, params=get_params)
        tasks = json.loads(get_response.text)
        return tasks

    def _make_observations(self, tasks):
        """Prepare observation set according agent API

        Args:
            :param tasks: dict object initialised with tasks JSON received from the testing system
            :type tasks: dict
        Returns:
            :return: list object containing observations in format, compatible with agent API
            :rtype: list
        """
        observations = []
        for task in tasks['qas']:
            observations.append({
                'id': task['id'],
                # Preprocess task
                'text': task['question'].split('\t')[0]
            })
        return observations

    def _get_predictions(self, observations):
        """Process observations with agent's model and get predictions on them

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
        Returns:
            :return: list object containing predictions (NER markup), extracted from each observation processing by agent
            :rtype: list
        """
        # Using agent.batch_act via feeding model 1 by 1 observation from batch
        predictions = []
        for observation in observations:
            prediction = self.agent.batch_act([observation])
            predictions.append({'id': prediction[0]['id'],'text': prediction[0]['text']})
        return predictions

    def _make_answers(self, observations, predictions):
        """Prepare answers dict for the JSON payload of the POST request

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
            :param predictions: list object containing predictions (NER markup)
            :type predictions: list
        Returns:
            :return: dict object containing answers to task, compatible with test system API for current KPI
            :rtype: dict
        """
        answers = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers[obs['id']] = pred['text']
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = answers
        return tasks

    def _get_score(self, answers):
        """Prepare POST request with answers, send to the KPI endpoint and get score

        Args:
            :param answers: dict object containing answers to task, compatible with test system API for current KPI
            :type answers: dict
        Returns:
            :return: dict object containing
                text: string with score information
                status_code: int with POST request response code
            :rtype: dict
        """
        post_headers = {'Accept': '*/*'}
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url'],
                                      json=answers,
                                      headers=post_headers)
        return {'text': rest_response.text, 'status_code': rest_response.status_code}

    def run_test(self, init_agent=True):
        """Rune full cycle of KPI testing sequence

        Args:
            :param init_agent: bool flag, turns on/off agent [re]initialising before testing sequence
            :type init_agent: bool
        """
        if init_agent or self.agent is None:
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

        score_response = self._get_score(answers)
        self.score = score_response['text']
        self.response_code = score_response['status_code']

    def run(self):
        while True:
            msg = self.input_queue.get()
            print(msg)
            self.run_test(init_agent=False)
            print("score %s" % self.score)
            self.output_queue.put(self.score)
