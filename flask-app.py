from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

data_storage= {}

@app.route('/send-data', methods=['POST'])
def receive_data():
    global data_storage
    data = request.json
    data_storage= data
    
    print("Received data:", data)
    return jsonify({"message": "Data received successfully"}), 200

@app.route('/get-data', methods=['GET'])
def send_data():
    return jsonify(data_storage), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run the Flask app
