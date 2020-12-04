import test as t
import flask
from flask import request, jsonify
import sqlite3
import time

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>API ONLINE</h1>'''

@app.route('/', methods=['POST'])
def test():
    start_time = time.time()

    result = {}
    data = request.form
    kalimat1 = data.get('kalimat1')
    kalimat2 = data.get('kalimat2')

    k1_preprocess = t.preprocess(kalimat1)
    k2_preprocess = t.preprocess(kalimat2)

    # k1_spellcheck = t.spell_check(k1_preprocess)
    # k2_spellcheck = t.spell_check(k2_preprocess)

    cosine1 = t.cosine_sim(k1_preprocess,k2_preprocess)

    qe = t.qe(k1_preprocess,k2_preprocess)

    cosine2 = t.cosine_sim(qe,k2_preprocess)
    # if qe != 'nc':
    # else:
    #     cosine2 = 'same'
    
    result['k1_preprocess'] = k1_preprocess
    result['k2_preprocess'] = k2_preprocess
    result['cosine1'] = cosine1
    result['qe'] = qe
    result['cosine2'] = cosine2
    result['time_est'] = time.time() - start_time
    # result['k1_spellcheck'] = k1_spellcheck
    # result['k2_spellcheck'] = k2_spellcheck
    return result

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

app.run()