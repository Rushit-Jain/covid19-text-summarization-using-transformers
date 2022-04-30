import subprocess
import nltk
import utilities
from flask import Flask, render_template, request

def initialize():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    command = ["python", "-m", "spacy", "download", "en_core_web_sm"]
    subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)

def process(paper,requested_entities):
    text = utilities.input_text(paper)
    sentences,sentencizer = utilities.sentencize(text)
    sentences_after_ner = utilities.load_ner_model(sentences,sentencizer)
    sentences_after_pos_tagging = utilities.pos_tagging(sentences_after_ner)
    sentences_after_synonym_substitution = utilities.synonym_substitution(sentences_after_pos_tagging)
    occurrences = utilities.entity_counting(sentences_after_synonym_substitution,requested_entities)
    summary = utilities.sentence_ranking(sentences,occurrences)
    return summary

app = Flask(__name__)
initialize()

@app.route("/home")
def form():
    return render_template('form_template.html')

@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    entities = ['GENE','DATE','QUANTITY','DISEASE','BODY_PART','PERSON','CHEMICAL','CELL','SPECIES','PROCEDURE','PLACE','PROCESS','PHENOMENON','VIRUS',]
    requested_entities = ""
    if request.method == 'POST':
        paper = request.files['paper']
        for entity in entities:
            if(request.form.get(entity.lower()) == "on"):
                requested_entities += " " + entity
        summary = process(paper,requested_entities)
        return render_template('output_template.html',summary=summary,entities=requested_entities.replace(" ",", ")[2:])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=3001)