import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from multitask_model import MultiTaskModel
import re

def preprocess_text(text):
    text = str(text).lower() 
    text = re.sub(r"[\d]+", "", text) 
    text = re.sub(r"[^\w\sঀ-৿.,!?₹$]", "", text)
    return text
    
# Set up the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

df = pd.read_csv("/mnt/partition1/machine_learning/Bengali_Sentiment_Analysis_and_Classification/dataset.csv")
df = df[["Review", "Sentiment", "Emotion"]] 
df.columns = ["text", "sentiment", "emotion"]
sentiment_encoder = LabelEncoder()
emotion_encoder = LabelEncoder()

df["sentiment"] = sentiment_encoder.fit_transform(df["sentiment"])
df["emotion"] = emotion_encoder.fit_transform(df["emotion"])

df["text"] = df["text"].apply(preprocess_text)


# Initialize the model (create a new instance of the model)
mbert_model = MultiTaskModel('bert-base-multilingual-uncased', df).to(device)
xlmr_model = MultiTaskModel('xlm-roberta-base', df).to(device)

# Load the trained model weights
mbert_model.load_state_dict(torch.load("mbert_model.pth"))
xlmr_model.load_state_dict(torch.load("xlmr_model.pth"))

# Initialize the optimizers (make sure the same learning rates are used)
optimizer_mbert = torch.optim.AdamW(mbert_model.parameters(), lr=1e-5)
optimizer_xlmr = torch.optim.AdamW(xlmr_model.parameters(), lr=1e-5)

# Load the optimizer states
optimizer_mbert.load_state_dict(torch.load("optimizer_mbert.pth"))
optimizer_xlmr.load_state_dict(torch.load("optimizer_xlmr.pth"))

# Load the tokenizers
mbert_tokenizer = AutoTokenizer.from_pretrained('./mbert_tokenizer')
xlmr_tokenizer = AutoTokenizer.from_pretrained('./xlmr_tokenizer')

# Set the models to evaluation mode if you want to perform inference
mbert_model.eval()
xlmr_model.eval()

# If you want to continue training (fine-tune), you can use the train_model function but you would need to import that module manually
# train_model(mbert_model, train_dataloader_mbert, optimizer_mbert, epochs=3)
# train_model(xlmr_model, train_dataloader_xlmr, optimizer_xlmr, epochs=3)

print("Models, optimizers, and tokenizers loaded successfully!")

def live_prediction(text):
    preprocessed_text = preprocess_text(text)

    # Tokenize for both models
    mbert_input = mbert_tokenizer(preprocessed_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    xlmr_input = xlmr_tokenizer(preprocessed_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    # Remove token_type_ids if they exist, because they are not needed in most cases
    mbert_input = {key: value for key, value in mbert_input.items() if key != 'token_type_ids'}
    xlmr_input = {key: value for key, value in xlmr_input.items() if key != 'token_type_ids'}

    # Predict with mBERT
    mbert_model.eval()
    with torch.no_grad():
        mbert_output = mbert_model(**mbert_input)
        mbert_sentiment_pred = torch.argmax(mbert_output[0], dim=1).cpu().numpy()[0]
        mbert_emotion_pred = torch.argmax(mbert_output[1], dim=1).cpu().numpy()[0]

    # Predict with XLM-R
    xlmr_model.eval()
    with torch.no_grad():
        xlmr_output = xlmr_model(**xlmr_input)
        xlmr_sentiment_pred = torch.argmax(xlmr_output[0], dim=1).cpu().numpy()[0]
        xlmr_emotion_pred = torch.argmax(xlmr_output[1], dim=1).cpu().numpy()[0]

    # Sentiment and Emotion Mapping
    sentiment_mapping = {0: "Negative", 1: "Positive"}
    emotion_mapping = {2:"Happy", 3:"Love", 4:"Sadness", 1:"Fear", 0:"Anger"}

    return sentiment_mapping[mbert_sentiment_pred], emotion_mapping[mbert_emotion_pred], sentiment_mapping[xlmr_sentiment_pred], emotion_mapping[xlmr_emotion_pred]

# To test the live prediction
while(True):
    user_input = input("Enter a product review for prediction: ")
    if user_input == "b" or user_input == "break":
        break
    mbert_sentiment, mbert_emoition, xlmr_sentiment, xlmr_emotion = live_prediction(user_input)
    print(f"mBERT Prediction: {mbert_sentiment} sentiment, {mbert_emoition} emotion")
    print(f"XLM-R Prediction: {xlmr_sentiment} sentiment, {xlmr_emotion} emotion")
    print('\n')
