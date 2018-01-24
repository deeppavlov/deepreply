from sanic import Sanic, request
from sanic.response import json, redirect
from sanic_openapi import swagger_blueprint, openapi_blueprint
from sanic_cors import CORS
from run_test import init_all_models


app = Sanic()
app.blueprint(openapi_blueprint)
app.blueprint(swagger_blueprint)
CORS(app)

models = None


@app.route('/')
async def index():
    return redirect('/apidocs/')


@app.route('/score', methods=['GET'])
async def score():
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
    if kpi_name not in ["kpi1", 'kpi2', "kpi3", "kpi4", "kpi4ru", "kpi3_2"]:
        return json({
            'error': 'kpi_name must be one of: kpi1, kpi2, kpi3, kpi4, "kpi4ru", "kpi3_2"'
        }), 400

    tasks_number = request.args.get('tasks_number')

    if not tasks_number.isdigit() or int(tasks_number) <= 0:
        return json({
            'error': 'tasks_number must be an integer, greater then zero'
        }), 400

    (model, in_q, out_q) = models[kpi_name]
    in_q.put(int(tasks_number))
    result = out_q.get()
    if isinstance(result, dict) and result.get("ERROR"):
        return json(result), 400

    return json(result), 200


@app.route('/answer/kpi1', methods=['POST'])
async def answer_kpi1():
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
async def answer_kpi2():
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
async def answer_kpi3():
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
async def answer_kpi3_2():
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


@app.route('/answer/kpi4', methods=['POST'])
async def answer_kpi4():
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


@app.route('/answer/kpi4ru', methods=['POST'])
async def answer_kpi4ru():
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


async def answer(kpi_name):
    if not request.is_json:
        return json({
            "error": "request must contains json data"
        }), 400

    text1 = request.get_json().get('text1') or ""
    text2 = request.get_json().get('text2') or ""

    if text1 == "":
        return json({
            "error": "request must contains non empty 'text1' parameter"
        }), 400

    (model, in_q, out_q) = await models[kpi_name]
    in_q.put([text1, text2])
    result = out_q.get()
    if isinstance(result, dict) and result.get("ERROR"):
        return json(result), 400
    return json(result), 200


if __name__ == "__main__":
    models = init_all_models()
    app.run(host='0.0.0.0', port=6001, debug=False)
