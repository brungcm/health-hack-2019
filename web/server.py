from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import threading
from random import randint

app = Flask(__name__)
api = Api(app)

CORS(app)

globvar = 0

@app.route("/")
def get():
    positionRandom = randint(0, 100)
    submergedRandom = randint(0, 300)
    panicRandom = randint(0, 100)
    return jsonify({ 'riskPosition': positionRandom, 'faceWaterSeconds': submergedRandom, 'riskPanic': panicRandom })

# class Employees(Resource):
#     def get(self):
#         return {'employees': [{'id':globvar, 'name':'Balram'},{'id':2, 'name':'Tom'}]} 

# class Employees_Name(Resource):
#     def get(self, employee_id):
#         print('Employee id:' + employee_id)
#         result = {'data': {'id':1, 'name':'Balram'}}
#         return jsonify(result)       


# api.add_resource(Employees, '/employees') # Route_1
# api.add_resource(Employees_Name, '/employees/<employee_id>') # Route_3

def mlFunction():
  threading.Timer(0.2, mlFunction).start()
  global globvar
  globvar += 1
  

mlFunction()

if __name__ == '__main__':
     app.run(port=5002)




