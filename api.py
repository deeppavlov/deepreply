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
    Run model for specified KPI on specified tasks number
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
    Run model for specified KPI on specified tasks number
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
    Run model for specified KPI on specified tasks number
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
    Run model for specified KPI on specified tasks number
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

    text1 = request.get_json().get('text1') or ""
    text2 = request.get_json().get('text2') or ""

    if text1 == "":
        return jsonify({
            "error": "request must contains non empty 'text1' parameter"
        }), 400

    (model, in_q, out_q) = models[kpi_name]
    in_q.put([text1, text2])
    result = out_q.get()

    return jsonify(result), 200


if __name__ == "__main__":
    models = init_all_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
