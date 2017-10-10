import os
import json
import requests

from parlai.core.params import ParlaiParser
from deeppavlov.agents.coreference.agents import CoreferenceAgent
from deeppavlov.tasks.coreference.utils import conll2dict


doc_path = '/home/madlit/Downloads/100.russian.v4_conll'
model_dir = '/home/madlit/Downloads/coreference/build'
model_file = 'model.max'

parser = ParlaiParser(True, True)
CoreferenceAgent.add_cmdline_args(parser)
args = ['-t deeppavlov.tasks.coreference.agents:BaseTeacher',
        '-m deeppavlov.agents.coreference.agents:CoreferenceAgent']
agent_params = parser.parse_args(args=args)
#agent_params['model_file'] = os.path.join(model_dir, model_file)
agent_params['model_file'] = model_dir
agent_params['datatype'] = 'test'
agent_params['language'] = 'russian'
agent_params['chosen_metric'] = 'conll-F-1'
agent_params['pretrained_model'] = True
agent_params['name'] = 'fasttext'

# with open(doc_path) as f: raw_doc = f.read()
doc = conll2dict(1, doc_path, 2, 3, 4)
print(doc)

agent = CoreferenceAgent(agent_params)
agent.observe(doc)
print('Pred: ')
pred = agent.predict()
print(pred)
