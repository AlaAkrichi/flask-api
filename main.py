from flask import Flask, request, jsonify
from flair.models import TextClassifier
from flair.data import Sentence
from flask_cors import CORS
import random
import pandas as pd

app = Flask(__name__)
CORS(app)
# Charger le modèle de classification de texte
classifier = TextClassifier.load('./ressources/final-model.pt')
df = pd.read_excel('reponse.xlsx')



def label_type(label):
    value = label.split('_')
    return value[-1]
@app.route('/classify', methods=['GET'])

def classify_text():
    question = request.args.get('question')
    print(question)
    # Créer une phrase à partir de la question
    sentence = Sentence(question)
    # Prédire la classe de la phrase
    classifier.predict(sentence)
    # Générer une réponse aléatoire de salutation
    filtered_df = df[df['label'] == label_type(sentence.labels[0].value)]
    random_item = filtered_df.sample()
    column_value = random_item['text'].values[0]
    # Construire la réponse JSON
    response = {"reponse": column_value}
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)