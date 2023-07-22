from flask import Flask, request, jsonify, render_template, make_response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import sys

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("DIAG-PSSeng/cicero-gpt2")
model = AutoModelForCausalLM.from_pretrained("DIAG-PSSeng/cicero-gpt2")

predictions = {}
current_id = 0

if __name__ == '__main__':
    # app.run(host="localhost", port=8000, debug=True)
    app.run(debug=True)

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/predictions', methods=['GET', 'POST', 'DELETE'])
def get_predictions():
    global predictions
    global current_id
    if request.method == 'GET':
        resp = make_response(jsonify(predictions), 200)
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.mimetype = 'application/json'
        return resp
    elif request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'text/plain'):
            input_request= request.data.decode('UTF-8')
            #DEBUG print(input_request, file=sys.stderr)

            # predict the result
            input_ids = tokenizer.encode(input_request, return_tensors='pt').to(device)
            output = model.generate(input_ids=input_ids, max_length=150, do_sample=True)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # save the prediction result
            current_id += 1
            predictions[current_id] = str(generated_text)
            #DEBUG print("L'id è: " + str(current_id), file=sys.stderr)
            #DEBUG print("La prediction è: " + str(generated_text), file=sys.stderr)
            #DEBUG print("Nel database c'è: " + predictions[current_id], file=sys.stderr)

            # send back the id to let the client inspect the result
            resp = make_response(str(current_id), 201)
            resp.headers.add('Access-Control-Allow-Origin', '*')
            resp.mimetype = 'text/plain'
            return resp

        else:
            return "Content type is not supported.", 400
        
    elif request.method == 'DELETE':
        predictions.clear()
        return "Predictions History Cleared", 200
    
    else:
        return "An error has occurred", 501


@app.route('/predictions/<prediction_id>', methods=['GET', 'DELETE'])
def handle_predictions(prediction_id):
    global predictions
    if request.method == 'GET':
        #if request.is_json:
            #return the predicted text of record "id"
            #DEBUG print("Al momento della get, Nel database c'è: " + predictions[int(prediction_id)], file=sys.stderr)
            resp = make_response(predictions[int(prediction_id)], 200)
            resp.headers.add('Access-Control-Allow-Origin', '*')
            resp.mimetype = 'text/plain'
            return resp

    elif request.method == 'DELETE':
        predictions.pop(prediction_id)
        return "The Prediction has been deleted" , 200
    
    else:
        return "An error has occurred", 501
