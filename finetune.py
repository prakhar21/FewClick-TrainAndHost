import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
from torch.utils.data import Dataset
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(sentence):
	tokens = tokenizer.tokenize(sentence)
	tokens = ['[CLS]'] + tokens + ['[SEP]']
	return tokens

def padding(tokens, max_len):
	if len(tokens) < max_len:
		tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
	else:
		tokens = tokens[:max_len-1] + ['[SEP]']
	return tokens

def attention_masks(tokens):
	attn_mask = [1 if token != '[PAD]' else 0 for token in tokens]
	return attn_mask

def numericalize(tokens):
	return tokenizer.convert_tokens_to_ids(tokens)

class CustomClassifierDataset(Dataset):
	def __init__(self, file, max_len, sep, col_txt, col_tag):
		self.dataframe = pd.read_csv(file, sep=sep)
		self.max_len = max_len
		self.sent = col_txt
		self.tag = col_tag

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		sentence = self.dataframe.loc[idx, self.sent]
		label = self.dataframe.loc[idx, self.tag]
		tokens = tokenize(sentence)
		pad_tokens = padding(tokens, self.max_len)
		tokens_ids = numericalize(pad_tokens)
		tokens_ids_tensor = torch.tensor(tokens_ids)
		attn_mask = (tokens_ids_tensor != 0).long()
		return tokens_ids_tensor, attn_mask, label

class CustomClassifier(nn.Module):

	def __init__(self, classes, freeze=True):
		super(CustomClassifier, self).__init__()
		
		self.bert = BertModel.from_pretrained('bert-base-uncased')
	   
		if freeze:
			for p in self.bert.parameters():
				p.requires_grad = False
		else:
			for p in self.bert.parameters():
				p.requires_grad = True
		
		self.dense = nn.Linear(768, classes)

	def forward(self, sequence, attn_masks):
		output, _ = self.bert(sequence, attention_mask=attn_masks)
		sent_vec = output[:, 0]
		logits = self.dense(sent_vec)
		return logits

def segment_data(data_file):
	try:
		import pandas as pd
	except ImportError:
		raise
	
	data = pd.read_csv(data_file, sep='\t', encoding='latin-1').sample(frac=1).drop_duplicates()
	data['sentence'] = data['sentence'].apply(lambda k: k.strip())
	classes = data['label'].unique()
	cleanup = {i: idx for idx, i in enumerate(classes)}
	data.replace(cleanup, inplace=True)

	data.iloc[0:int(len(data)*0.8)].to_csv('./mydata/train.tsv', sep='\t', index = False, header = True)
	data.iloc[int(len(data)*0.8):].to_csv('./mydata/dev.tsv', sep='\t', index = False, header = True)
	return classes


def val_custom(net, criterion, valloader):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in valloader:
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits, labels).item()
            count += 1

    return mean_loss / count

def prepare_data(data):
	classes = segment_data(data)
	no_classes = len(classes)
	
	train_set = CustomClassifierDataset(file='./mydata/train.tsv', max_len=30, sep='\t', col_txt='sentence', col_tag='label')
	val_set = CustomClassifierDataset(file='./mydata/dev.tsv', max_len=30, sep='\t', col_txt='sentence', col_tag='label')

	train_loader = DataLoader(train_set, batch_size=128, num_workers=3)
	val_loader = DataLoader(val_set, batch_size=128, num_workers=3)

	return train_loader, val_loader, no_classes

def train_custom(data):
	epoch = 1

	train_loader, val_loader, no_classes = prepare_data(data)
	net = CustomClassifier(no_classes)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=2e-1)

	for ep in range(epoch):
		for idx, (tokens, attn_masks, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {ep+1}'):
			optimizer.zero_grad()
			logits = net(tokens, attn_masks)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

		val_loss = val_custom(net, criterion, val_loader)
		print (f"Training Loss : {loss.item()} -- Validation Loss : {val_loss}")
	return