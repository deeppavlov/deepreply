from flask import Flask, request, jsonify, redirect
from flasgger import Swagger

from run_test import init_all_models

app = Flask(__name__)
Swagger(app)

models = None


@app.before_first_request
def __init_models():
    global models
    models = init_all_models()

@app.route('/')
def index():
    return redirect('/apidocs/')


@app.route('/run_kpi', methods=['GET'])
def answer():
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
    if kpi_name not in ["kpi1", 'kpi2', "kpi3", "kpi4", "kpi11"]:
        return jsonify({
            'error': 'kpi_name must be one of: kpi1, kpi2, kpi3, kpi4, kpi11'
        }), 500

    tasks_number = request.args.get('tasks_number')

    if not tasks_number.isdigit() or int(tasks_number) <= 0:
        return jsonify({
            'error': 'tasks_number must be an integer, greater then zero'
        }), 500

    (model, in_q, out_q) = models[kpi_name]
    in_q.put(tasks_number)
    result = out_q.get()

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
