from flask import Flask, request, jsonify
from flask_cors import CORS
from infer import search

app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    text = data["text"]

    images = search(text)
    return jsonify(images)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
