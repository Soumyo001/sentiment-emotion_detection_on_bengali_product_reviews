from transformers import AutoModelForSequenceClassification
import torch

class MultiTaskModel(torch.nn.Module):
    def __init__(self, model_name, df):
        super(MultiTaskModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, output_hidden_states=True)  # Enable hidden states output
        self.sentiment_head = torch.nn.Linear(self.model.config.hidden_size, 2)  # Sentiment output head
        self.emotion_head = torch.nn.Linear(self.model.config.hidden_size, len(df['emotion'].unique()))  # Emotion output head

    def forward(self, input_ids, attention_mask):
        # Pass the input through the transformer model
        outputs = self.model(input_ids, attention_mask=attention_mask)  # This gives logits and hidden states
        
        # Extract hidden states (the last hidden state of the model)
        hidden_states = outputs.hidden_states[-1]  # The last hidden state
        
        # Extract the first token representation (CLS token)
        cls_hidden_state = hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Get sentiment and emotion logits from the heads
        sentiment_logits = self.sentiment_head(cls_hidden_state)
        emotion_logits = self.emotion_head(cls_hidden_state)
        
        return sentiment_logits, emotion_logits


