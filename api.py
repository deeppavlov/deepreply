from flask import Flask, request, jsonify, redirect
from flasgger import Swagger

from run_test import init_all_models

app = Flask(__name__)
Swagger(app)

models = None


@app.route('/')
def index():
    return redirect('/apidocs/')


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


def answer(kpi_name):
    if not request.is_json:
        return jsonify({
            "error": "request must contains json data"
        }), 400

    text0 = request.get_json().get('text0') or ""
    text1 = request.get_json().get('text1') or ""
    text2 = request.get_json().get('text2') or ""

    if text0 == "":
        if text1 == "":
            return jsonify({
                "error": "request must contains non empty 'text0 or text1' parameter"
            }), 400
        else:
            q_put = [text1, text2]
    else:
        if isinstance(text0, int):
            q_put = text0
        else:
            return jsonify({
                "error": "'text0' parameter must be integer number"
            }), 400

    (model, in_q, out_q) = models[kpi_name]
    in_q.put(q_put)
    result = out_q.get()

    return jsonify(result), 200


if __name__ == "__main__":
    models = init_all_models()
    app.run(host='127.0.0.1', port=5000, debug=False)
