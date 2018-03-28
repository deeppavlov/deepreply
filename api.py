from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS
from run_test import init_all_models


TWO_ARGUMENTS_MODELS = ['kpi2', 'kpi4', 'kpi4ru', 'kpi4en']


app = Flask(__name__)
Swagger(app)
CORS(app)

models = None


@app.route('/')
def index():
    return redirect('/apidocs/')


@app.route('/score', methods=['GET'])
def score():
    """
    Run model for specified KPI on specified tasks number
    ---
    parameters:
      - name: kpi_name
        in: query
        required: true
        type: string
      - name: tasks_number
        in: query
        required: true
        type: string
    """
    kpi_name = request.args.get('kpi_name')
    if kpi_name not in ["kpi1", 'kpi2', "kpi3", "kpi4", "kpi4ru",
                        "kpi3_2", "kpi3en", "intents", "kpi4en",
                        "ranking_en", "ner_en_ontonotes", "odqa_en"]:
        return jsonify({
            'error': 'kpi_name must be one of: kpi1, kpi2, kpi3, kpi4, '
                     'kpi4ru, kpi3_2, kpi3en, intents, kpi4en,'
                     'ranking_en, "ner_en_ontonotes", "odqa_en"'
        }), 400

    tasks_number = request.args.get('tasks_number')

    if not tasks_number.isdigit() or int(tasks_number) <= 0:
        return jsonify({
            'error': 'tasks_number must be an integer, greater then zero'
        }), 400

    (model, in_q, out_q) = models[kpi_name]
    in_q.put(int(tasks_number))
    result = out_q.get()
    if isinstance(result, dict) and result.get("ERROR"):
        return jsonify(result), 400

    return jsonify(result), 200


@app.route('/answer/kpi1', methods=['POST'])
def answer_kpi1():
    """
    KPI 1: Insults
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi1")


@app.route('/answer/kpi2', methods=['POST'])
def answer_kpi2():
    """
    KPI 2: Paraphraser
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi2")


@app.route('/answer/kpi3', methods=['POST'])
def answer_kpi3():
    """
    KPI 3: NER
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi3")


@app.route('/answer/kpi3_2', methods=['POST'])
def answer_kpi3_2():
    """
    KPI 3: NER (By Le Ahn)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi3_2")


@app.route('/answer/kpi3en', methods=['POST'])
def answer_kpi3en():
    """
    KPI 3: NER (English)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi3en")


@app.route('/answer/kpi4', methods=['POST'])
def answer_kpi4():
    """
    KPI 4: SQUAD
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi4")


@app.route('/answer/kpi4en', methods=['POST'])
def answer_kpi4en():
    """
    KPI 4: SQUAD (English)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi4en")


@app.route('/answer/kpi4ru', methods=['POST'])
def answer_kpi4ru():
    """
    KPI 4: SQUAD (Russian)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("kpi4ru")


@app.route('/answer/intents', methods=['POST'])
def answer_intents():
    """
    KPI: Intents Recogintion
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("intents")


@app.route('/answer/ranking_en', methods=['POST'])
def answer_ranking_en():
    """
    KPI: Ranking (English)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("ranking_en")


@app.route('/answer/ner_en_ontonotes', methods=['POST'])
def answer_ner_en_ontonotes():
    """
    KPI: NER ontonotes (English)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("ner_en_ontonotes")


@app.route('/answer/odqa_en', methods=['POST'])
def odqa_en():
    """
    KPI: ODQA (English)
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("odqa_en")


@app.route('/answer/proxy', methods=['POST'])
def proxy():
    """
    Web pages proxy
    ---
    parameters:
     - name: data
       in: body
       required: true
       type: json
    """
    return answer("proxy")


def answer(kpi_name):
    if not request.is_json:
        return jsonify({
            "error": "request must contains json data"
        }), 400

    text1 = request.get_json().get('text1') or ""
    text2 = request.get_json().get('text2') or ""

    if text1.strip() == "":
        return jsonify({
            "error": "request must contain non empty 'text1' parameter"
        }), 400

    if kpi_name in TWO_ARGUMENTS_MODELS and text2.strip() == "":
        return jsonify({
            "error": "request must contain non empty 'text2' parameter"
        }), 400

    (model, in_q, out_q) = models[kpi_name]
    in_q.put([text1, text2])
    result = out_q.get()
    if isinstance(result, dict) and result.get("ERROR"):
        return jsonify(result), 400
    return jsonify(result), 200


if __name__ == "__main__":
    models = init_all_models()
    app.run(host='0.0.0.0', port=6001, threaded=True)
