from flask import Flask, request, Response, jsonify
from Summary_Test.test import Summary
from Lexical_Simplification.test import predict

app = Flask(__name__)

summary_obj = Summary()

summary_text = ""
@app.route('/summary', methods=['GET', 'POST'])
def summary():
    global summary_text
    data = request.form['ctext']
    summary_text = summary_obj.getSummary(str(data))

    final_result = {"Original Text": str(data), "Summary Text": str(summary_text)}
    return jsonify(final_result)

@app.route('/final', methods=['GET', 'POST'])
def lexical():
    global summary_text
    lexical_text = predict(summary_text)

    final_result = {"Original Text": str(summary_text), "Lexical Simplification Text": str(lexical_text)}
    return jsonify(final_result)    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)