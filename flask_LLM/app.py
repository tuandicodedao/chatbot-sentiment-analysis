from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Đường dẫn đến mô hình đã huấn luyện và tokenizer
model_path = 'bert_model'  # Escape dấu gạch chéo ngược
tokenizer_path = 'tokenizer_bert_model'

# Tải mô hình và tokenizer từ thư mục đã lưu
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Hàm dự đoán nhãn cảm xúc
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# API dự đoán cảm xúc
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if text:
        sentiment = predict_sentiment(text)
        sentiment_label = 'Negative' if sentiment == 0 else 'Positive'
        return jsonify({'sentiment': sentiment_label})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
