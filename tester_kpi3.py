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
import copy

import build_utils as bu
from parlai.core.agents import create_agent
from tester_base import TesterBase


class TesterKpi3(TesterBase):
    """Implements methods of TesterBase for testing KPI3 with NER agent

    Public methods:
        init_agent(self): initiates model agent
    """

    def __init__(self, config, opt, input_queue, output_queue):
        """Tester class constructor

        Args:
            :param config: dict object initialised with config.json and modified with run script
            :type config: dict
            :param opt: dict object with optional agent and KPI testing parameters
            :type opt: dict
            :param opt:
            :type opt: multiprocessing.Queue
            :param opt:
            :type opt: multiprocessing.Queue
        All params a needed to init base class (TesterBase) instance
        """
        super(TesterKpi3, self).__init__(config, opt, input_queue, output_queue)

    def init_agent(self):
        """Initiate model agent

        Implementation of base class TesterBase abstract method
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

    def _make_observations(self, tasks, human_input=False):
        """Prepare observation set according agent API

        Args:
            :param tasks: dict object initialised with tasks JSON received from the testing system
            :type tasks: dict
        Returns:
            :return: list object containing observations in format, compatible with agent API
            :rtype: list
        Implementation of base class TesterBase abstract method
        """
        observations = []
        if human_input:
            observations.append({
                'id': 'dummy',
                # Preprocess task
                'text': tasks[0].split('\t')[0]
            })
        else:
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
        Implementation of base class TesterBase abstract method
        """
        # Using agent.batch_act via feeding model 1 by 1 observation from batch
        predictions = []
        for observation in observations:
            prediction = self.agent.batch_act([observation])
            predictions.append({'id': prediction[0]['id'],'text': prediction[0]['text']})
        return predictions

    def _make_answers(self, observations, predictions, human_input=False):
        """Prepare answers dict for the JSON payload of the POST request

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
            :param predictions: list object containing predictions (NER markup)
            :type predictions: list
        Returns:
            :return: dict object containing answers to task, compatible with test system API for current KPI
            :rtype: dict
        Implementation of base class TesterBase abstract method
        """
        answers = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers[obs['id']] = pred['text']
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = answers
        if human_input:
            return answers['dummy']
        else:
            return tasks
