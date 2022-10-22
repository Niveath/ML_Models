from IPython.display import clear_output
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import spacy
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import tqdm

import warnings
warnings.filterwarnings('ignore')

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

!unzip /content/smsspamcollection.zip
!rm /content/readme
!rm !rm /content/smsspamcollection.zip

clear_output()

"""download the GloVe embeddings database
   I am using word embeddings with 300 dimensions"""

!wget https://nlp.stanford.edu/data/glove.6B.zip

!unzip /content/glove.6B.zip

!rm -rf /content/glove.6B.zip
!rm /content/glove.6B.50d.txt
!rm /content/glove.6B.100d.txt
!rm /content/glove.6B.200d.txt

clear_output()

text = []
label = []

with open("/content/SMSSpamCollection") as f:

    """ read each line of the text file and create a Pandas Data Frame
        label spam messages as 1 and legit messages as 0"""

    lines = f.readlines()
    for line in lines:
        l = len(line)
        if(line[0] == "h"):
             label.append(0)
             text.append(line[4:l - 2])
        else:
            label.append(1)
            text.append(line[5:l - 2])

!rm /content/SMSSpamCollection

sms = pd.DataFrame(zip(text, label), columns = ["Text", "Label"])
sms['Text_Length'] = sms["Text"].apply(lambda x: len(x.split()))

spacy_tokenizer = spacy.load('en_core_web_sm')
def tokenize (text):

    """remove any non-ascii characters
       remove punctuations
       tokenize the text"""

    text = re.sub(r"[^\x00-\x7F]+", "", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub("", text.lower())
    tokenized_text = [token.text.lower() for token in spacy_tokenizer.tokenizer(text)]
    return tokenized_text

word_counts = Counter()

sms["Tokenized_Text"] = sms["Text"].apply(lambda x: tokenize(x))

for i in range(len(sms)):
    tokenized_text = sms.loc[i, "Tokenized_Text"]
    word_counts.update(tokenized_text)

# sms.head()

def load_GloVe_embeddings():

    """load the GloVe embeddings from the files downloaded"""

    embeddings_dict_GloVe = {}

    with open("/content/glove.6B.300d.txt", 'r', encoding = 'utf-8') as GloVe:
        for line in GloVe:
            word_vector = line.split()
            word = word_vector[0]
            vector = np.asarray(word_vector[1:], dtype = "float32")
            embeddings_dict_GloVe[word] = vector
    
    return embeddings_dict_GloVe

def get_embeddings(pretrained_vectors, word_counts, embedding_size = 300):

    """get the embeddings of the words in the dataset"""

    vocab_size = len(word_counts) + 2
    vocab = ["", "UNK"]

    vocab_to_index = {}
    vocab_to_index["PAD"] = 0
    vocab_to_index["UNK"] = 1

    embeddings = np.empty([vocab_size, embedding_size])

    embeddings[0] = np.zeros(embedding_size)
    embeddings[1] = np.random.uniform(-1, 1, embedding_size)


    for index, word in enumerate(word_counts.keys()):
        if word in pretrained_vectors:
            embeddings[index+2] = pretrained_vectors[word]
        else:
            embeddings[index+2] = embeddings[1]
        vocab_to_index[word] = index + 2
        vocab.append(word)

    vocab = np.array(vocab)
    
    return embeddings, vocab, vocab_to_index

word_vectors = load_GloVe_embeddings()

word_embeddings, vocab, vocab_to_index = get_embeddings(word_vectors, word_counts, embedding_size = 300)

def embed_text(tokenized_text, word_embeddings, vocab_to_index, max_text_length=20, embedding_size = 300):
    text_length = len(tokenized_text)

    embedding = [word_embeddings[vocab_to_index["PAD"]]]
    embedding *= max_text_length
    embedding = np.asarray(embedding)

    if(text_length>max_text_length):
        text_length = max_text_length

    for i in range(text_length):
        word = tokenized_text[i]
        if word in vocab_to_index:
            embedding[i, :] = word_embeddings[vocab_to_index[word]]
        else:
            embedding[i, :] = word_embeddings[vocab_to_index["UNK"]]

    return embedding

sms["Embedded Vector"] = sms["Tokenized_Text"].apply(lambda x: embed_text(x, word_embeddings, vocab_to_index))

X = list(sms["Embedded Vector"])
y = list(sms["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

class load_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.float32)), self.y[idx]

train_data = load_dataset(X_train, y_train)
test_data = load_dataset(X_test, y_test)

batch_size = 32
train_set = DataLoader(train_data, batch_size = batch_size, shuffle = False)
test_set = DataLoader(test_data, batch_size = batch_size)

def train_model(model, train_set, validation_set, device, epochs=10, lr=3e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_set:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            pred = []
            for x in y_pred:
                if(x>=0.5):
                     pred.append(1)
                else:
                    pred.append(0)
            pred = torch.as_tensor(pred).to(device)
            correct += (pred == y).float().sum()
            optimizer.zero_grad()
            loss = loss_function(y_pred, torch.reshape(y, (-1, 1)).float())
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        train_loss = sum_loss/total
        train_acc = correct/total
        val_loss, val_acc = validation_metrics(model, device, validation_set)

        print(f"Epoch: {epoch+1:02} Train Loss: {train_loss:.4f} | Training Accuracy: {train_acc * 100:.4f}| Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.4f}")

def validation_metrics (model, device, validation_set):
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()
    
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y in validation_set:
        x = x.to(device)
        y_pred = model(x)
        y = y.to(device)
        loss = loss_function(y_pred, torch.reshape(y, (-1, 1)).float())
        pred = []
        for x in y_pred:
            if(x>=0.5):
                    pred.append(1)
            else:
                pred.append(0)
        pred = torch.as_tensor(pred).to(device)
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total, correct/total

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, num_of_layers, word_embeddings, dropout = 0.2, embedding_dim = 300, bidirectional = True):
        super(LSTM, self).__init__()

        #since we are using pre-trained GloVe embeddings, there is no need for a nn.Embedding layer

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim, 
                            num_layers=num_of_layers,
                            batch_first=True,
                            dropout=dropout, 
                            bidirectional=self.bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)

        self.fully_connected_layer = nn.Linear(hidden_dim * 2, output_dim) if(bidirectional) else nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        output, (hidden_state, cell_state) = self.lstm(text)

        if(self.bidirectional):
            hidden_state = torch.cat((hidden_state[-2, :, :], hidden_state[-1, : , :]), dim = 1)
        else:
            hidden_state = hidden_state[-1]
        
        outputs = self.fully_connected_layer(hidden_state)
        
        return outputs

vocab_size = len(vocab)
hidden_dim = 512
output_dim = 1
num_of_layers = 2
dropout = 0.2
embedding_dim = 300

lstm_model = LSTM(vocab_size, hidden_dim, output_dim, num_of_layers, word_embeddings, dropout, embedding_dim)

epochs = 20
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

train_model(lstm_model, train_set, test_set, device, epochs, learning_rate)