# Fine-tuning BERT and RoBERTa for Tweet Sentiment Classification (Sentiment140 dataset)
# The code was originally run on a Jupyter Notebook in Google Colab (NVIDIA A100 GPU)

# The script below presents the essential functions
# Each section could be a separate python script (you will have to handle the imports and data pre-processing)

### 0 - Imports and Colab configuration

from google.colab import drive
drive.mount('/content/drive')

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from peft import LoraConfig, get_peft_model, TaskType

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
device = "cuda" if torch.cuda.is_available() else "cpu"

### 1 - Data pre-processing and analysis
TRAIN_DATA = '/content/drive/MyDrive/sentiment140/sentiment140_traindata.csv'
TEST_DATA = '/content/drive/MyDrive/sentiment140/sentiment140_testdata.csv'

columns = ["target", "ids", "date", "flag", "user", "text"]
train_dataset = pd.read_csv(TRAIN_DATA, encoding='latin-1',names=columns)
test_dataset = pd.read_csv(TEST_DATA, encoding='latin-1',names=columns)

train_dataset = train_dataset[['ids','target','text']] # we don't need the other columns
test_dataset = test_dataset[['ids','target','text']]

train_dataset['target'] = train_dataset['target'].map({0:0, 4:1}) # changing the original labels (0=negative, 1=positive, 2=neutral)
test_dataset['target'] = test_dataset['target'].map({0:0, 4:1, 2:2})

#print(train_dataset.shape, test_dataset.shape)
#print(train_dataset.head()) 
#print(train_dataset.info())
#print(test_dataset.head())  

## Distribution of sentiment classes
train_dataset['target'].value_counts().plot(kind='bar',color='royalblue')
plt.title("Distribution of Sentiment Classes - Train set - 1.6M rows")
plt.xticks(rotation=0)
plt.xlabel("Sentiment (0=negative, 1=positive)")
plt.ylabel("Nb. of tweets")
plt.show()

test_dataset['target'].value_counts().plot(kind='bar',color='royalblue')
plt.title("Distribution of Sentiment Classes - Test set - 498 rows")
plt.xticks(rotation=0)
plt.xlabel("Sentiment (0=negative, 1=positive, 2=neutral)")
plt.ylabel("Nb. of tweets")
plt.show()

## Tweet length analysis
train_dataset['text_len'] = train_dataset['text'].str.len()
train_dataset['text_len'].hist(bins=50,color='royalblue')
plt.title("Tweet Length Distribution - Train set")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()
train_dataset.groupby('target')['text_len'].mean() # average lenght / sentiment (train)

test_dataset['text_len'] = test_dataset['text'].str.len()
test_dataset['text_len'].hist(bins=50,color='royalblue')
plt.title("Tweet Length Distribution - Test set")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()
test_dataset.groupby('target')['text_len'].mean() # average lenght / sentiment (test)

## word/token analysis + wordcloud
def get_most_common_words(df, label, n=100):
    words = " ".join(df[df['target']==label]['text']).lower()
    words = re.findall(r'\b\w+\b', words)
    return Counter(words).most_common(n)

#print("Top Negative Words:", get_most_common_words(train_dataset, 0))
#print("Top Positive Words:", get_most_common_words(train_dataset, 1))

text_neg = " ".join(train_dataset[train_dataset['target']==0]['text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(text_neg)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Tweets")
plt.show()

text_pos = " ".join(train_dataset[train_dataset['target']==1]['text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(text_pos)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Tweets")
plt.show()

### 2 - Processing/Tokenization
def clean_tweet(text):
    text = text.lower()                           
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'@\w+', '', text)              
    text = re.sub(r'#', '', text)                 
    text = re.sub(r'\s+', ' ', text).strip()     
    return text

train_dataset['clean_text'] = train_dataset['text'].apply(clean_tweet) #to lowercase tweets, remove urls, remove mentions, hashtags, extra whitespaces

## Tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # BERT tokenizer
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base") # RoBERTa tokenizer

train_hf = Dataset.from_pandas(train_dataset[['clean_text', 'target']]) #converting panda dataframe to a suitable format for HF dataframes
#train_hf[0]

### 2.1 - BERT tokenization
def tokenize_with_bert(batch):
    return bert_tokenizer(
        batch["clean_text"], 
        padding="max_length", #to ensure equal length
        truncation=True, 
        max_length=128 
    )

train_enc_bert = train_hf.map(tokenize_with_bert, batched=True)
save_path_bert_tokenized = "/content/drive/MyDrive/sentiment140/train_bert_tokenized"
train_enc_bert.save_to_disk(save_path_bert_tokenized)
print(f"Dataset saved to {save_path_bert_tokenized}")

decoded_text = bert_tokenizer.decode(train_enc_bert[0]["input_ids"], skip_special_tokens=True)
#print(decoded_text)

## Test BERT
example_bert = train_enc_bert[0]
input_ids_bert = example_bert["input_ids"]
attention_mask_bert = example_bert["attention_mask"]
decoded_bert_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids_bert)
for inp, token, mask in zip(input_ids_bert, decoded_bert_tokens, attention_mask_bert):
    print(f"input_ids : {inp} | {token:12} | attention_mask : {mask} ")
print("Input IDs:", input_ids_bert)
print("Attention mask:", attention_mask_bert)
print("Decoded:", bert_tokenizer.decode(input_ids_bert))

### 2.2 - RoBERTa tokenization
def tokenize_with_roberta(batch):
    return roberta_tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_enc_roberta = train_hf.map(tokenize_with_roberta, batched=True)

save_path_roberta_tokenized = "/content/drive/MyDrive/sentiment140/train_roberta_tokenized"
train_enc_roberta.save_to_disk(save_path_roberta_tokenized)
print(f"Dataset saved to {save_path_roberta_tokenized}")

## Test RoBERTa
example_roberta = train_enc_roberta[0]
input_ids_roberta = example_roberta["input_ids"]
attention_mask_roberta = example_roberta["attention_mask"]
decoded_roberta_tokens = roberta_tokenizer.convert_ids_to_tokens(input_ids_roberta)
for inp, token, mask in zip(input_ids_roberta, decoded_roberta_tokens, attention_mask_roberta):
    print(f"input_ids : {inp} | {token:12} | attention_mask : {mask} ")
print("Input IDs:", input_ids_roberta)
print("Attention mask:", attention_mask_roberta)
print("Decoded:", roberta_tokenizer.decode(input_ids_roberta))

### 3 - Embeddings
### 3.1 -  BERT embeddings
train_enc_bert = load_from_disk("/content/drive/MyDrive/sentiment140/train_bert_tokenized") #we load the tokenized bert
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

sampled_tweets = train_enc_bert.shuffle(seed=42).select(range(160000)) #10%

## Embeddings extraction
def get_cls_embeddings(dataset, model, batch_size=32):
    all_embs, all_labels = [], []
    for i in range(0, len(dataset), batch_size):
        print(i)
        batch = dataset[i:i+batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] token ; <s> token in RoBERTa
        all_embs.append(cls_emb)
        all_labels.extend(batch["target"])
    return torch.cat(all_embs, dim=0), all_labels

def get_mean_embeddings(dataset, model, batch_size=32):
    all_embs, all_labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            mean_emb = torch.sum(last_hidden * mask_expanded, 1) / mask_expanded.sum(1)
        all_embs.append(mean_emb.cpu())
        all_labels.extend(batch["target"])
    return torch.cat(all_embs, dim=0), all_labels

## embeddings extraction
cls_embs, cls_labels = get_cls_embeddings(sampled_tweets, bert_model)
mean_embs, mean_labels = get_mean_embeddings(sampled_tweets, bert_model)
print("CLS embeddings shape:", cls_embs.shape)
print("Mean embeddings shape:", mean_embs.shape)

## Visualization of clusters (dim reduction with PCA and t-SNE)
def plot_embeddings(embeddings, labels, title=""):
    reduced = PCA(n_components=50).fit_transform(embeddings)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

    labels = np.array(labels)  
    colors = {0: "red", 1: "blue"}
    names = {0: "Negative", 1: "Positive"}

    plt.figure(figsize=(10,10))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(
            tsne_2d[idx, 0], tsne_2d[idx, 1],
            c=colors[cls], label=names[cls], alpha=0.6, s=20
        )
    plt.title(title)
    plt.legend()
    plt.show()

plot_embeddings(cls_embs.numpy(), cls_labels, title="BERT CLS Embeddings (160k tweets)")
plot_embeddings(mean_embs.numpy(), mean_labels, title="BERT Mean-Pooled Embeddings (160k tweets)")

### 3.2 - RoBERTA embeddings
train_enc_roberta = load_from_disk("/content/drive/MyDrive/sentiment140/train_roberta_tokenized")

roberta_model = AutoModel.from_pretrained("roberta-base").to(device)
roberta_model.eval()

sampled_tweets_roberta = train_enc_roberta.shuffle(seed=42).select(range(160000))

cls_embs_roberta, cls_labels_roberta = get_cls_embeddings(sampled_tweets_roberta, roberta_model) # Note : function named get_cls_embeddings() but it does extract <s> for RoBERTa
mean_embs_roberta, mean_labels_roberta = get_mean_embeddings(sampled_tweets_roberta, roberta_model)
print("RoBERTa CLS embeddings shape:", cls_embs_roberta.shape)
print("RoBERTa Mean embeddings shape:", mean_embs_roberta.shape)

## Visualization of clusters
plot_embeddings(cls_embs_roberta.numpy(), cls_labels_roberta, title="RoBERTa <s> Embeddings (160k tweets)")
plot_embeddings(mean_embs_roberta.numpy(), mean_labels_roberta, title="RoBERTa Mean-Pooled Embeddings (160k tweets)")

### 4 - Baseline models (Pre-trained BERT/RoBERTA + Linear Classifier vs TF-IDF + LogReg)
### 4.1 - Pre-trained BERT + Linear Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
    
X = mean_embs  
y = torch.tensor(mean_labels) 

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

input_dim = X.shape[1]  # 768 for BERT base
model = SimpleClassifier(input_dim).to(device)


## training/validation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40
best_f1 = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}, "
          f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

## final evaluation
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n===== Best Model Evaluation =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

### 4.2 - Pre-trained RoBERTa + Linear Classifier
X = mean_embs_roberta 
y = torch.tensor(mean_labels_roberta) 

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

input_dim = X.shape[1]  
model = SimpleClassifier(input_dim).to(device)

## training/validaiotn
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40
best_f1 = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}, "
          f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

## final evaluation
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n===== Best Model Evaluation =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

### 4.3 - TF-IDF + Logistic Regression
sampled_subset = train_hf.shuffle(seed=42).select(list(range(160000)))
sampled_dataframe = sampled_subset.to_pandas()

texts = sampled_dataframe["clean_text"]
labels = np.array(sampled_dataframe["target"])  # make sure numpy is imported

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

## tf-idf vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

## logreg classifier
clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_val_tfidf)

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("\nTF-IDF + Logistic Regression")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

## confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - TF-IDF + LogReg")
plt.show()



### 5 - Full Fine-tuning 
### 5.1 - Full Fine-tuing BERT
train_enc_bert = load_from_disk("/content/drive/MyDrive/sentiment140/train_bert_tokenized")
train_enc_bert = train_enc_bert.rename_column("target", "labels")
train_enc_bert = train_enc_bert.shuffle(seed=42).select(range(160000))  # 160k sample
train_enc_bert = train_enc_bert.train_test_split(test_size=0.2, seed=42)
train_enc_bert["validation"] = train_enc_bert.pop("test")

bert_base_uncased = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

training_args_bert_finetuning = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

trainer_bert_finetuning = Trainer(
    model=bert_base_uncased,
    args=training_args_bert_finetuning,
    train_dataset=train_enc_bert["train"],
    eval_dataset=train_enc_bert["validation"],
    compute_metrics=compute_metrics,
)

## training
trainer_bert_finetuning.train()
results = trainer_bert_finetuning.evaluate()
print("Eval results:", results)

save_path_finetuned_bert = "/content/drive/MyDrive/sentiment140/bert_finetuned_model"
trainer_bert_finetuning.save_model(save_path_finetuned_bert)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(save_path_finetuned_bert)

## embeddings extraction
bert_base_uncased.eval()
val_dataset = train_enc_bert["validation"]
input_ids = torch.tensor(val_dataset["input_ids"])
attention_mask = torch.tensor(val_dataset["attention_mask"])
labels = torch.tensor(val_dataset["labels"])
val_loader = DataLoader(TensorDataset(input_ids, attention_mask, labels), batch_size=32, shuffle=False)

all_embs, all_labels = [], []
with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
        batch_input_ids, batch_attention_mask = batch_input_ids.to(device), batch_attention_mask.to(device)
        outputs = bert_base_uncased.bert(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = batch_attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_emb = torch.sum(last_hidden * mask_expanded, 1) / mask_expanded.sum(1)
        all_embs.append(mean_emb.cpu())
        all_labels.extend(batch_labels.numpy())

embeddings = torch.cat(all_embs).numpy()
labels = np.array(all_labels)

## visualization
reduced = PCA(n_components=50, random_state=42).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

plt.figure(figsize=(10,10))
colors = ["red", "blue"]
for cls in [0, 1]:
    idx = labels == cls
    plt.scatter(tsne_2d[idx,0], tsne_2d[idx,1], c=colors[cls], label=["Negative","Positive"][cls], alpha=0.1, s=20)
plt.legend()
plt.title("BERT Fine-tuned Mean-Pooled Embeddings (160k tweets)")
plt.show()

### 5.2 - Full Fine-tuing RoBERTa
train_enc_roberta = load_from_disk("/content/drive/MyDrive/sentiment140/train_roberta_tokenized")
train_enc_roberta = train_enc_roberta.rename_column("target", "labels")
train_enc_roberta = train_enc_roberta.shuffle(seed=42).select(range(160000))  # 160k sample
train_enc_roberta = train_enc_roberta.train_test_split(test_size=0.2, seed=42)
train_enc_roberta["validation"] = train_enc_roberta.pop("test")

roberta_base = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

training_args_roberta_finetuning = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

trainer_roberta_finetuning = Trainer(
    model=roberta_base,
    args=training_args_roberta_finetuning,
    train_dataset=train_enc_roberta["train"],
    eval_dataset=train_enc_roberta["validation"],
    compute_metrics=compute_metrics,
)

## training
trainer_roberta_finetuning.train()
results = trainer_roberta_finetuning.evaluate()
print("Eval results:", results)

save_path_finetuned_roberta = "/content/drive/MyDrive/sentiment140/roberta_finetuned_model"
trainer_roberta_finetuning.save_model(save_path_finetuned_roberta)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.save_pretrained(save_path_finetuned_roberta)

## embeddings
roberta_base.eval()
val_dataset = train_enc_roberta["validation"]
input_ids = torch.tensor(val_dataset["input_ids"])
attention_mask = torch.tensor(val_dataset["attention_mask"])
labels = torch.tensor(val_dataset["labels"])
val_loader = DataLoader(TensorDataset(input_ids, attention_mask, labels), batch_size=32, shuffle=False)

all_embs, all_labels = [], []
with torch.no_grad():
    for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
        batch_input_ids, batch_attention_mask = batch_input_ids.to(device), batch_attention_mask.to(device)
        outputs = roberta_base.roberta(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = batch_attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_emb = torch.sum(last_hidden * mask_expanded, 1) / mask_expanded.sum(1)
        all_embs.append(mean_emb.cpu())
        all_labels.extend(batch_labels.numpy())

embeddings = torch.cat(all_embs).numpy()
labels = np.array(all_labels)

## visualization
reduced = PCA(n_components=50, random_state=42).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

plt.figure(figsize=(8,6))
colors = ["red", "blue"]
for cls in [0, 1]:
    idx = labels == cls
    plt.scatter(tsne_2d[idx,0], tsne_2d[idx,1], c=colors[cls], label=["Negative","Positive"][cls], alpha=0.1, s=20)
plt.legend()
plt.title("RoBERTa Fine-tuned Mean-Pooled Embeddings (160k tweets)")
plt.show()

### 7 - Fine-tuning BERT and RoBERTa with LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  
    r=16,                        # rank
    lora_alpha=32,               
    lora_dropout=0.1,            
    target_modules=["query", "value"]  #Q, V
)

### 7.1 - Fine-tuning BERT with LoRA
train_enc_bert = load_from_disk("/content/drive/MyDrive/sentiment140/train_bert_tokenized")
train_enc_bert = train_enc_bert.rename_column("target", "labels")
train_enc_bert = train_enc_bert.shuffle(seed=42).select(range(160000))  # 160k sample
train_enc_bert = train_enc_bert.train_test_split(test_size=0.2, seed=42)
train_enc_bert["validation"] = train_enc_bert.pop("test")

bert_base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

lora_wrapped_bert_base_model = get_peft_model(bert_base_model, lora_config).to(device)
print(lora_wrapped_bert_base_model) # where lora is injected

bert_lora_training_args = TrainingArguments(
    output_dir="./results_lora",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,  # usually slightly higher for LoRA
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

bert_lora_trainer = Trainer(
    model=lora_wrapped_bert_base_model,
    args=bert_lora_training_args,
    train_dataset=train_enc_bert["train"],
    eval_dataset=train_enc_bert["validation"],
    compute_metrics=compute_metrics,
)

## fine-tune
bert_lora_trainer.train()
results = bert_lora_trainer.evaluate()
print("Eval results (LoRA):", results)

save_bert_lora = "/content/drive/MyDrive/sentiment140/bert_lora_model"
bert_lora_trainer.save_model(save_bert_lora)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(save_bert_lora)

## embeddeings
lora_wrapped_bert_base_model.eval()
val_dataset = train_enc_bert["validation"]

input_ids = torch.tensor(val_dataset["input_ids"]).to(device)
attention_mask = torch.tensor(val_dataset["attention_mask"]).to(device)
labels = torch.tensor(val_dataset["labels"]).numpy()
val_loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=32, shuffle=False)

all_embs = []
with torch.no_grad():
    for batch_input_ids, batch_attention_mask in val_loader:
        # encoder, not classification head
        outputs = lora_wrapped_bert_base_model.bert(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            return_dict=True
        )

        # last_hidden = [batch, seq_len, hidden_dim]
        last_hidden = outputs.last_hidden_state  

        # mean-pooling
        mask_expanded = batch_attention_mask.unsqueeze(-1).expand(last_hidden.size()).float() 
        mean_emb = torch.sum(last_hidden * mask_expanded, 1) / mask_expanded.sum(1)

        all_embs.append(mean_emb.cpu().numpy())

embeddings = np.vstack(all_embs)

## visualization
reduced = PCA(n_components=50, random_state=42).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

plt.figure(figsize=(10,10))
colors = ["red", "blue"]
for cls in [0, 1]:
    idx = labels == cls
    plt.scatter(tsne_2d[idx,0], tsne_2d[idx,1], 
                c=colors[cls], label=["Negative","Positive"][cls], 
                alpha=0.1, s=20)
plt.legend()
plt.title("BERT + LoRA Fine-tuned Embeddings (160k tweets)")
plt.show()

### 7.2 - Fine-tuning RoBERTa with LoRA
train_enc_roberta = load_from_disk("/content/drive/MyDrive/sentiment140/train_roberta_tokenized")
train_enc_roberta = train_enc_roberta.rename_column("target", "labels")
train_enc_roberta = train_enc_roberta.shuffle(seed=42).select(range(160000))  # 160k sample
train_enc_roberta = train_enc_roberta.train_test_split(test_size=0.2, seed=42)
train_enc_roberta["validation"] = train_enc_roberta.pop("test")

roberta_base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=2)

lora_wrapped_roberta_base_model = get_peft_model(roberta_base_model, lora_config).to(device)
print(lora_wrapped_roberta_base_model)

robert_lora_training_args = TrainingArguments(
    output_dir="./results_lora",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,  # usually slightly higher for LoRA
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

roberta_lora_trainer = Trainer(
    model=lora_wrapped_roberta_base_model,
    args=robert_lora_training_args,
    train_dataset=train_enc_roberta["train"],
    eval_dataset=train_enc_roberta["validation"],
    compute_metrics=compute_metrics,
)

roberta_lora_trainer.train()
results = roberta_lora_trainer.evaluate()
print("📊 Eval results (LoRA):", results)

save_roberta_lora = "/content/drive/MyDrive/sentiment140/roberta_lora_model"
roberta_lora_trainer.save_model(save_roberta_lora)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.save_pretrained(save_roberta_lora)


lora_wrapped_roberta_base_model.eval()
val_dataset = train_enc_roberta["validation"]

input_ids = torch.tensor(val_dataset["input_ids"]).to(device)
attention_mask = torch.tensor(val_dataset["attention_mask"]).to(device)
labels = torch.tensor(val_dataset["labels"]).numpy()

val_loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=32, shuffle=False)

all_embs = []
with torch.no_grad():
    for batch_input_ids, batch_attention_mask in val_loader:
        outputs = lora_wrapped_roberta_base_model.roberta(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            return_dict=True
        )

        last_hidden = outputs.last_hidden_state  

        mask_expanded = batch_attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_emb = torch.sum(last_hidden * mask_expanded, 1) / mask_expanded.sum(1)

        all_embs.append(mean_emb.cpu().numpy())

embeddings = np.vstack(all_embs)

## visualization
reduced = PCA(n_components=50, random_state=42).fit_transform(embeddings)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

plt.figure(figsize=(10,10))
colors = ["red", "blue"]
for cls in [0, 1]:
    idx = labels == cls
    plt.scatter(tsne_2d[idx,0], tsne_2d[idx,1], 
                c=colors[cls], label=["Negative","Positive"][cls], 
                alpha=0.1, s=20)
plt.legend()
plt.title("RoBERTa + LoRA Fine-tuned Embeddings (32k tweets)")
plt.show()

### 8 - Ultimate test : New tweets sentiment prediction with LoRa-fine-tuned RoBERTa

test_tweets = [
    # Positive tweets
    "I just got a promotion at work, feeling amazing!",
    "This new phone is fantastic, I love it!",
    "Had a wonderful dinner with my family tonight",
    "Feeling grateful for all the good things in life",
    "The movie was hilarious, I can't stop laughing",

    # Positive tweets that look negative
    "Finally finished all my work, now I can relax (so exhausted though!)",
    "Missed the bus but hey, I got to enjoy a beautiful walk",
    "I have so many tasks today, but I feel motivated",
    "Long day at the gym, but I feel stronger than ever",
    "Had a stressful meeting, but it went better than I feared",
    
    # Negative tweets
    "I'm so tired of all this bad news",
    "Missed my train and now I'm late, terrible day",
    "Feeling sick and miserable today",
    "I hate when my plans get ruined",
    "This weather is depressing and gloomy",

    # Negative tweets that look positive
    "Great, another Monday… just what I needed",
    "I won a free ticket, but I still feel miserable",
    "Finally done with chores, now I get to be bored",
    "My friend is visiting, yet I feel drained",
    "I got praised at work, still can’t shake off the anxiety"
]

tokenized_tweets = tokenizer(test_tweets, padding=True, truncation=True, return_tensors="pt").to(device)

lora_wrapped_roberta_base_model.eval()
with torch.no_grad():
    outputs = lora_wrapped_roberta_base_model(**tokenized_tweets)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

for tweet, pred in zip(test_tweets, preds):
    label = "Positive" if pred == 1 else "Negative"
    print(f"[{label}] {tweet}")