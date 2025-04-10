import re
import os
import torch
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from torch.nn.functional import sigmoid
from transformers import BertTokenizer, BertForSequenceClassification

# Suppress BeautifulSoup warning
from bs4 import MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class BERT_Tagger:
    def __init__(self):
        print("Initializing BERT_Tagger...")
        self.model = None
        self.tokenizer = None
        self.label_binarizer = None
        self.tag_list = []

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"<code>.*?</code>", "[CODE]", text, flags=re.DOTALL)
        return text.strip()

    def train(self, df, top_n_tags=1000, num_epochs=3, save_path='./saved_models/bert_tagger.pkl'):
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.model_selection import train_test_split
        from transformers import Trainer, TrainingArguments

        print("Training BERT Tagger...")

        # Rename columns to expected names
        df = df.rename(columns={"Title": "title", "Question": "body", "Tags": "tags"})

        # Convert tags column into list of tags
        df['tags'] = df['tags'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

        # Keep only top N most frequent tags
        all_tags = pd.Series([tag for tags in df['tags'] for tag in tags])
        top_tags = set(all_tags.value_counts().head(top_n_tags).index)
        df['tags'] = df['tags'].apply(lambda tags: [tag for tag in tags if tag in top_tags])
        df = df[df['tags'].map(len) > 0].copy()

        if df.empty:
            raise ValueError("No data left after filtering low-frequency tags.")

        # Clean title and body, and merge
        df['text'] = df.apply(
            lambda row: self.clean_text(row['title']) + ' ' + self.clean_text(row['body']),
            axis=1
        )

        # Binarize tags for multi-label classification
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df['tags'])
        self.tag_list = list(mlb.classes_)
        self.label_binarizer = mlb

        # Tokenize the cleaned text
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer
        texts = list(df['text'])
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        labels = y

        # Dataset class for PyTorch
        class TagDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.float32)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        # Split into train and val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        train_dataset = TagDataset(train_encodings, train_labels)
        val_dataset = TagDataset(val_encodings, val_labels)

        # Load BERT with correct number of labels
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.tag_list),
            problem_type="multi_label_classification"
        )
        self.model = model

        # Training configuration with live logs
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_strategy="epoch",                 # ✅ Updated from deprecated `evaluation_strategy`
            logging_dir='./logs',
            logging_strategy="steps",              # ✅ Log every few steps
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            disable_tqdm=False,                    # ✅ Enable live progress bar
            logging_first_step=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        self.save_model(save_path)

    def predict(self, text, top_k=5):
        cleaned_text = self.clean_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = sigmoid(logits)[0]
            top_indices = torch.topk(probs, k=top_k).indices.tolist()
            return [self.tag_list[i] for i in top_indices]

    def save_model(self, file_path):
        print(f"Saving BERT model to {file_path}...")
        weight_path = file_path.replace('.pkl', '_weights.pt')
        torch.save(self.model.state_dict(), weight_path)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'tag_list': self.tag_list,
                'label_binarizer': self.label_binarizer
            }, f)
        print("Model saved successfully.")

    def load_model(self, file_path):
        print(f"Loading BERT model from {file_path}...")
        weight_path = file_path.replace('.pkl', '_weights.pt')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.tokenizer = data['tokenizer']
        self.tag_list = data['tag_list']
        self.label_binarizer = data['label_binarizer']

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.tag_list),
            problem_type="multi_label_classification"
        )
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        print("Model loaded successfully.")
