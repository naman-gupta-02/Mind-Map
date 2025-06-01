from flask import Flask, render_template, request, flash
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import os
import gdown
import zipfile

app = Flask(__name__)
app.secret_key = "your_secret_key"

model_path = "bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)


# Define labels
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        if not prompt.strip():
            flash("Prompt cannot be empty.")
        else:
            # Tokenize and predict
            inputs = tokenizer(prompt, return_tensors="tf", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
            label = id2label[prediction]
            flash(f"Predicted Sentiment: {label}")
    return render_template("index.html", prompt=prompt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5020))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port)
