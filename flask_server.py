import redis
from flask import Flask, jsonify, Response
from flask_cors import CORS

r = redis.StrictRedis(host='localhost', port=6379, db=0)

app = Flask(__name__)
CORS(app)

@app.route('/get', methods=['GET'])
def hello():
    rtv = {'hand': str(r.get('foo'))}
    print rtv
    return jsonify(rtv)
    #return Response(jsonify(rtv), mimetype='application/json')


if __name__ == '__main__':
    app.run(port=9998, debug=True)