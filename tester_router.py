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


import requests
import traceback

from tester_base import TesterBase


class TesterRouter(TesterBase):
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
        super(TesterRouter, self).__init__(config, opt, input_queue, output_queue)

    def init_agent(self):
        """Initiate model agent

        Implementation of base class TesterBase abstract method
        """
        pass

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
        pass

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
        pass

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
        pass

    def _route_post(self, post_payload):
        post_headers = {'Accept': 'application/json'}
        payload = {'text1': post_payload[0]}
        payload['text2'] = post_payload[1] if len(post_payload) > 1 else ''
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url_post'],
                                      json=payload,
                                      headers=post_headers)
        return rest_response.json()

    def _route_get(self, numtasks):
        rest_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url_get']
        get_headers = {'Accept': 'application/json'}
        payload = {'tasks_number': numtasks}
        rest_response = requests.get(rest_url,
                                     params=payload,
                                     headers=get_headers)
        return rest_response.json()

    def run(self):
        while True:
            try:
                input_q = self.input_queue.get()
                print("Run %s, received input: %s" % (self.kpi_name, str(input_q)))
                if isinstance(input_q, list):
                    print("%s human input mode..." % self.kpi_name)
                    result = self._route_post(input_q)
                    print("%s action result:  %s" % (self.kpi_name, result))
                    self.output_queue.put(result)
                elif isinstance(input_q, int):
                    print("%s API mode..." % self.kpi_name)
                    result = self._route_get(input_q)
                    print("%s action result:  %s" % (self.kpi_name, result))
                    self.output_queue.put(result)
                else:
                    self.output_queue.put({"ERROR":
                                               "{} parameter error - {} belongs to unknown type".format(self.kpi_name,
                                                                                                        str(input_q))})
            except Exception as e:
                self.output_queue.put({"ERROR": "{}".format(traceback.extract_stack())})
