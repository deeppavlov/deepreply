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


class Tester(TesterBase):
    """Implements methods of TesterBase for testing KPI4 with SQUAD agent

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
        super(Tester, self).__init__(config, opt, input_queue, output_queue)

    def init_agent(self):
        """Initiate model agent

        Implementation of base class TesterBase abstract method
        """
        params = ['-t', 'squad',
                    '-m', 'deeppavlov.agents.squad.squad:SquadAgent',
                    '--batchsize', '64',
                    '--display-examples', 'False',
                    '--num-epochs', '-1',
                    '--log-every-n-secs', '60',
                    '--log-every-n-epochs', '-1',
                    '--validation-every-n-secs', '1800',
                    '--validation-every-n-epochs', '-1',
                    '--chosen-metrics', 'f1',
                    '--validation-patience', '5',
                    '--type', 'fastqa_default',
                    '--linear_dropout', '0.0',
                    '--embedding_dropout', '0.5',
                    '--rnn_dropout', '0.0',
                    '--recurrent_dropout', '0.0',
                    '--input_dropout', '0.0',
                    '--output_dropout', '0.0',
                    '--context_enc_layers', '1',
                    '--question_enc_layers', '1',
                    '--encoder_hidden_dim', '300',
                    '--projection_dim', '300',
                    '--pointer_dim', '300',
                    '--datatype', 'test']
        opt = bu.arg_parse(params)
        embeddings_dir = self.config['embeddings_dir']
        embedding_file = self.config['kpis'][self.kpi_name]['settings_agent']['embedding_file']
        dict_file = self.config['kpis'][self.kpi_name]['settings_agent']['dict_files_names']
        model_files = self.opt['model_files']
        opt['model_file'] = model_files[0]
        opt['pretrained_model'] = model_files[0]
        opt['dict_file'] = os.path.join(os.path.dirname(model_files[0]), dict_file)
        if self.opt['embedding_file'] is not None:
            opt['embedding_file'] = self.opt['embedding_file']
        else:
            opt['embedding_file'] = os.path.join(embeddings_dir, embedding_file)

        self.agent = create_agent(opt)

    def _make_observations(self, tasks):
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
        for task in tasks['paragraphs']:
            for question in task['qas']:
                observations.append({
                    'id': question['id'],
                    'text': '%s\n%s' % (task['context'], question['question'])})
        return observations

    def _batchfy_observations(self, observations, batch_length):
        """Make batch of batches from observation: split observations set to several batches

        Args:
            :param observations: list object containing observations in format, compatible with agent API
            :type observations: list
            :param batch_length: int number containing number of observations in one sub-batch
            :type batch_length: int
        Returns:
            :return: list object containing lists of observations with number elements <= batch_length value
            :rtype: list
        """
        return [observations[i:i + batch_length] for i in range(0, len(observations), batch_length)]

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
        observations_batchsize = int(self.config['kpis'][self.kpi_name]['settings_kpi']['observations_batchsize'])
        predictions = []
        if observations_batchsize == 0:
            predictions = self.agent.batch_act(observations)
        elif observations_batchsize > 0:
            # Split batch of observations for several batches and process observations via algorithm
            predictions = []
            observ_batch = self._batchfy_observations(observations, observations_batchsize)
            for observ in observ_batch:
                predict = self.agent.batch_act(observ)
                predictions.extend(predict)
        return predictions

    def _make_answers(self, observations, predictions):
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
            answers[obs['id']] = pred['text']
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = answers
        return tasks
