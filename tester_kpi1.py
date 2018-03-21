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
from deeppavlov.tasks.insults.build import data_preprocessing


class TesterKpi1(TesterBase):
    """Implements methods of TesterBase for testing KPI1 with Insults agent

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
        super(TesterKpi1, self).__init__(config, opt, input_queue, output_queue)

    def init_agent(self):
        """Initiate model agent

        Implementation of base class TesterBase abstract method
        """
        params = ['-t', 'deeppavlov.tasks.insults.agents:FullTeacher',
            '-m', 'deeppavlov.agents.insults.insults_agents:EnsembleInsultsAgent',
            '--model_coefs', '0.3333333', '0.3333333', '0.3333334',
            '--datatype', 'test',
            '--batchsize', '64',
            '--display-examples', 'False',
            '--max_sequence_length', '100',
            '--filters_cnn', '256',
            '--kernel_sizes_cnn', '1 2 3',
            '--embedding_dim', '100',
            '--dense_dim', '100']
        embeddings_dir = self.config['embeddings_dir']
        embedding_file = self.config['kpis'][self.kpi_name]['settings_agent']['embedding_file']
        model_files = self.opt['model_files']
        opt = bu.arg_parse(params)
        opt['model_files'] = model_files
        opt['model_names'] = self.config['kpis'][self.kpi_name]['settings_agent']['model_names']
        opt['raw_dataset_path'] = os.path.dirname(model_files[0])
        if self.opt['embedding_file'] is not None:
            opt['fasttext_model'] = self.opt['embedding_file']
        else:
            opt['fasttext_model'] = os.path.join(embeddings_dir, embedding_file)

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
                'text': self._preprocess_input(tasks[0]),
            })
        else:
            for task in tasks['qas']:
                observations.append({
                    'id': task['id'],
                    'text': self._preprocess_input(task['question']),
                })
        return observations

    def _preprocess_input(self, input):
        input_list = [input]
        preprocessed_list = data_preprocessing(input_list)
        preprocessed_input = preprocessed_list[0]
        return preprocessed_input

    def _get_predictions(self, observations):
        """Process observations with agent's model and get predictions on them

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
        Returns:
            :return: list object containing predictions in raw agent format
            :rtype: list
        Implementation of base class TesterBase abstract method
        """
        predictions = self.agent.batch_act(observations)
        return predictions

    def _make_answers(self, observations, predictions, human_input=False):
        """Prepare answers dict for the JSON payload of the POST request

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
            :param predictions: list object containing predictions in raw agent format
            :type predictions: list
        Returns:
            :return: dict object containing answers to task, compatible with test system API for current KPI
            :rtype: dict
        Implementation of base class TesterBase abstract method
        """
        answers = {}
        observ_predict = list(zip(observations, predictions))
        for obs, pred in observ_predict:
            answers[obs['id']] = pred['score']
        if human_input:
            return answers['dummy']
        else:
            tasks = copy.deepcopy(self.tasks)
            tasks['answers'] = answers
            return tasks
