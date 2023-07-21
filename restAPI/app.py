from flask import Flask, request, jsonify, render_template, make_response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("DIAG-PSSeng/cicero-gpt2")
model = AutoModelForCausalLM.from_pretrained("DIAG-PSSeng/cicero-gpt2")

predictions = {}
current_id = 0

if __name__ == '__main__':
    # app.run(host="localhost", port=8000, debug=True)
    # reset variables at server start/restart
    predictions.clear()
    current_id = 0
    app.run(debug=True)

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/predictions', methods=['GET', 'POST', 'DELETE'])
def get_predictions():
    if request.method == 'GET':
        resp = make_response(jsonify(predictions), 200)
        resp.mimetype = 'application/json'
        return resp
    elif request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'text/plain'):
            input_request= str(request.get_data);

            # predict the result
            input_ids = tokenizer.encode(input_request, return_tensors='pt').to(device)
            output = model.generate(input_ids=input_ids, max_length=150, do_sample=True)

            # save the prediction result
            predictions[++current_id] = output
            # send back the id to let the client inspect the result
            return current_id, 201

        else:
            return "Content type is not supported.", 400
        
    elif request.method == 'DELETE':
        predictions.clear()
        return "Predictions History Cleared", 200
    
    else:
        return "An error has occurred", 501


@app.route('predictions/<prediction_id>', methods=['GET', 'DELETE'])
def handle_predictions(prediction_id):
    if request.method == 'GET':
        if request.is_json:
            #return the predicted text of record "id"
            return predictions[prediction_id], 200

    elif request.method == 'DELETE':
        predictions.pop(prediction_id)
        return "The Prediction has been deleted" , 200
    
    else:
        return "An error has occurred", 501
