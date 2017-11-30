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
import re
import copy

import build_utils as bu
from parlai.core.agents import create_agent
from tester_base import TesterBase


class Tester(TesterBase):
    """Implements methods of TesterBase for testing KPI11 with Paraphraser agent

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
        if self.opt['embedding_file'] is not None:
            opt['embeddings_path'] = self.opt['embedding_file']
        else:
            opt['embeddings_path'] = os.path.join(embeddings_dir, embedding_file)

        self.agent = create_agent(opt)

    def _make_observations(self, tasks):
        """Prepare observation set according agent API

        Args:
            :param tasks: dict object initialised with tasks JSON received from the testing system
            :type tasks: dict
        Returns:
            :return: dict object containing observations in format, compatible with agent API
            :rtype: dict
        Implementation of base class TesterBase abstract method
        """
        id = []
        observs = []
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
            observation = {'conll': [], 'valid_conll': [conll_str.split('\n')], 'id': ''}
            id.append(task['id'])
            observs.append(observation)
        observations = {'id': id, 'observation': observs}
        return observations

    def _extract_coref(self, conll):
        """Extract coreference markup from conll formatted text

        Args:
            :param conll: list object containing strings of conll text with coreference markup
            :type conll: list
        Returns:
            :return: str object containing coreference markup of conll formatted text
            :rtype: str
        """
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

    def _get_predictions(self, observations):
        """Process observations with agent's model and get predictions on them

        Args:
            :param observations: dict object containing observations in format, compatible with agent API
            :type observations: dict
        Returns:
            :return: list object containing predictions in raw agent format
            :rtype: list
        Implementation of base class TesterBase abstract method
        """
        predictions = []
        for observation in observations['observation']:
            self.agent.observe(observation)
            prediction = self.agent.act()
            predictions.append(prediction['valid_conll'][0])
        return predictions

    def _make_answers(self, observations, predictions):
        """Prepare answers dict for the JSON payload of the POST request

        Args:
            :param observations: dict object containing observations in format, compatible with agent API
            :type observations: dict
            :param predictions: list object containing predictions in raw agent format
            :type predictions: list
        Returns:
            :return: dict object containing answers to task, compatible with test system API for current KPI
            :rtype: dict
        Implementation of base class TesterBase abstract method
        """
        id_predict = {}
        observe_predict = list(zip(observations['id'], predictions))
        for obs, pred in observe_predict:
            id_predict[obs] = self._extract_coref(''.join(pred))
        tasks = copy.deepcopy(self.tasks)
        tasks['answers'] = id_predict
        return tasks
