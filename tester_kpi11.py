import os
import json
import requests
import re

import build_utils as bu
from parlai.core.agents import create_agent


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

    # Initiate agent
    def init_agent(self):
        params = ['-t', 'deeppavlov.tasks.coreference_scorer_model.agents:CoreferenceTeacher',
                    '-m', 'deeppavlov.agents.coreference_scorer_model.agents:CoreferenceAgent',
                    '--display-examples', 'False',
                    '--num-epochs', '20',
                    '--log-every-n-secs', '-1',
                    '--log-every-n-epochs', '1',
                    '--validation-every-n-epochs', '1',
                    '--chosen-metrics', 'f1',
                    '--validation-patience', '20',
                    #'--language', 'russian',
                    '--datatype', 'test']
        opt = bu.arg_parse(params)
        opt['model_file'] = '/home/madlit/github/sbertest/build/models/kpi11_1'
        opt['pretrained_model'] = '/home/madlit/github/sbertest/build/models/kpi11_1'
        opt['embeddings_filename'] = 'ft_0.8.3_nltk_yalen_sg_300.bin'
        print(opt)
        self.agent = create_agent(opt)

    # Update Tester config with or without [re]initiating agent
    def update_config(self, config, init_agent=False):
        self.config = config
        if init_agent:
            self.init_agent()

    # Update tasks number
    def set_numtasks(self, numtasks):
        self.numtasks = numtasks

    # Get kpi11 tasks via REST
    def _get_tasks(self):
        get_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url']
        if self.numtasks in [None, 0]:
            test_tasks_number = self.config['kpis'][self.kpi_name]['settings_kpi']['test_tasks_number']
        else:
            test_tasks_number = self.numtasks
        get_params = {'stage': 'test', 'quantity': test_tasks_number}
        get_response = requests.get(get_url, params=get_params)
        tasks = json.loads(get_response.text)
        #with open('/home/madlit/Downloads/cored_doc.json') as json_data:
        #    tasks = json.load(json_data)
        return tasks

    # Prepare observations set
    def _make_observations(self, tasks):
        observations = []
        for task in tasks['qas']:
            conll_str = str(task['question'])
            doc_num = str(re.search(r'#begin document [(].+[)];\n([0-9]+)', conll_str).group(1))
            conll_str = re.sub(r'(?P<subst>#begin document [(].+[)];)',
                               '#begin document(%s); part 0' % doc_num,
                               conll_str)
            match = re.search(r'\n\n#end document', conll_str)
            if match is None:
                conll_str = re.sub(r'\n#end document', r'\n\n#end document', conll_str)
            print(conll_str)
            observation = {
                'conll': [],
                'valid_conll': [conll_str.split('\n')],
                'id': ''
            }
            observations.append({
                'id': task['id'],
                'observation': observation
            })
        return observations

    def _extract_coref(self, conll):
        coref_str = ''
        lines = conll.split('\n')
        #lines = conll['conll_str'].split('\n')
        for i in range(len(lines)):
            if lines[i].startswith("#begin"):
                #continue
                coref_str += ' '
            elif lines[i].startswith("#end document"):
                #continue
                coref_str += ' '
            else:
                row = lines[i].split('\t')
                if len(row) == 1:
                    #continue
                    coref_str += ' '
                else:
                    coref_str += row[-1] + ' '
        return coref_str

    # Process observations via algorithm
    def _get_predictions(self, observations):
        predictions = {}
        for observation in observations:
            self.agent.observe(observation['observation'])
            prediction = self.agent.act()
            #predictions[observation['id']] = self._extract_coref(prediction)
            predictions[observation['id']] = self._extract_coref(''.join(prediction['valid_conll'][0]))
            print(predictions[observation['id']])
        return predictions

    # Generate answers data
    def _make_answers(self, predictions):
        answers = {}
        answers['id'] = self.tasks['id']
        answers['answers'] = predictions
        return answers
        #tasks = self.tasks
        #tasks['answers'] = predictions
        #return tasks

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
        print('Tasks:')
        print(tasks)

        observations = self._make_observations(tasks)
        self.observations = observations
        print('Observations:')
        print(observations)

        predictions = self._get_predictions(observations)
        self.predictions = predictions
        print('Predictions:')
        print(predictions)

        answers = self._make_answers(predictions)
        self.answers = answers
        print('answers:')
        print(json.dumps(answers))

        score = self._get_score(answers)
        self.score = score
        print('score:')
        print(score)
