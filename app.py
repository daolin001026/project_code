from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import csv
import os
import re

app = Flask(__name__)

MODEL_PATH = 'mbti_bert_model'
TOKENIZER_PATH = 'mbti_bert_tokenizer'

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

labels = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
          "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

OUTPUT_FILE = 'results/mbti_predictions.csv'

def encode_text(text, tokenizer):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

def predict(text, model, tokenizer, device, label_encoder):
    inputs = encode_text(text, tokenizer)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).flatten()

    return label_encoder.inverse_transform(preds.cpu().numpy())[0]

def load_user_data(file_path):
    user_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                user_data[row[0]] = row[2]
    return user_data

def check_share_info(share_text, user_data):
    involved_users = [user for user in user_data if re.search(r'\b' + re.escape(user) + r'\b', share_text)]
    if not involved_users:
        return ["No users found in the share text."]

    suggestions = []
    for user in involved_users:
        mbti_type = user_data[user]
        if mbti_type.startswith('I'):
            suggestions.append(f"Do not recommend sharing because {user} is {mbti_type}.")
        else:
            suggestions.append(f"Recommend sharing because {user} is {mbti_type}.")
    return suggestions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        user_name = request.form['user_name']
        input_text = request.form['input_text']
        predicted_type = predict(input_text, model, tokenizer, device, label_encoder)

        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_name, input_text, predicted_type])
        
        return render_template('predict.html', user_name=user_name, input_text=input_text, predicted_type=predicted_type, success=True)
    return render_template('predict.html')

@app.route('/share', methods=['GET', 'POST'])
def share_page():
    if request.method == 'POST':
        share_text = request.form['share_text']
        user_data = load_user_data(OUTPUT_FILE)
        suggestions = check_share_info(share_text, user_data)
        return render_template('share.html', share_text=share_text, suggestions=suggestions)
    return render_template('share.html')

if __name__ == '__main__':
    app.run(debug=True)