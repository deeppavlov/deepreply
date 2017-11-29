from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger


import os

from run_test import main as run_kpi


app = Flask(__name__)
Swagger(app)


@app.route('/')
def index():
    return redirect('/apidocs/')


@app.route('/run_kpi', methods=['GET'])
def answer():
    """
    Run model for specified KPI on specified tusks number
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
    tasks_number = request.args.get('tasks_number')

    run_kpi(['-k', kpi_name, '-t', tasks_number])

    result = {
        'kpi_name': kpi_name,
        'tasks_number': tasks_number,
        'result': 'ok'
    }

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)