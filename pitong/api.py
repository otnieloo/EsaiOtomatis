import test as t
import flask
from flask import request, jsonify
import sqlite3
import time
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>API ONLINE</h1>'''

@app.route('/', methods=['POST'])
def main():
    start_time = time.time()

    result = {}
    data = request.form
    kalimat1 = data.get('kalimat1')
    kalimat2 = data.get('kalimat2')

    k1_preprocess = t.preprocess(kalimat1)
    k2_preprocess = t.preprocess(kalimat2)

    # k1_spellcheck = t.spell_check(k1_preprocess)

    cosine1_notf = t.cosine_sim(k1_preprocess,k2_preprocess)

    cosine1 = t.tfIdfCosine(k1_preprocess,k2_preprocess)

    qe = t.qe(k1_preprocess,k2_preprocess)

    cosine2_notf = t.cosine_sim(qe,k2_preprocess)
    
    cosine2 = t.tfIdfCosine(qe,k2_preprocess)
    
    k1_preprocess = t.preprocess(kalimat1)

    match = []

    for i in qe:
        if i not in k1_preprocess:
            match.append(i)

    # print(match)
    result['k1_raw'] = kalimat1
    result['k2_raw'] = kalimat2
    result['k1_preprocess'] = k1_preprocess
    result['k2_preprocess'] = k2_preprocess
    # result['k1_spellcheck'] = k1_spellcheck
    result['cosine1'] = cosine1
    result['cosine2'] = cosine2
    result['cosine1_notf'] = cosine1_notf
    result['cosine2_notf'] = cosine2_notf
    
    result['qe'] = qe
    result['match'] = match
 
    result['time_est'] = time.time() - start_time

    return json.dumps(result)

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

app.run()