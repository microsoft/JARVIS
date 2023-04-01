import argparse
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS
from awesome_chat import chat_huggingface
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()


config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
httpserver = config["httpserver"]
host = httpserver["host"]
port = httpserver["port"]

app = flask.Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

@app.route('/hugginggpt', methods=['POST'])
def chat():
    data = request.get_json()
    # print(data)
    messages = data["messages"]
    response = chat_huggingface(messages)
    return jsonify(response)

if __name__ == '__main__':
    waitress.serve(app, host=host, port=port)
    # app.run(host=host, port=port, debug=True)