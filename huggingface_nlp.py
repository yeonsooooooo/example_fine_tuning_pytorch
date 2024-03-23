from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

checkpoint = "bert-base-uncased"

#tokenizer 정의
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#데이터셋 로드
raw_datasets = load_dataset("glue", "mrpc")

"""
수정부분 시작
 - 모델에 들어가는 데이터셋 형태를 잘 맞춰 주어야 함.
 - lable -> labels로 수정
 - train_data와 valid_data를 함수로 변경하거나, TestDataset 클래스의 __init__에 정의해도 됨.
"""
#데이터셋 전처리
train_data = {'input_ids': [],
              'token_type_ids': [],
              'attention_mask': [],
              'labels': []}

for i in range(len(raw_datasets['train'])):
  tokenize = tokenizer(raw_datasets['train']['sentence1'][i], raw_datasets['train']['sentence2'][i], truncation=True)
  train_data['input_ids'].append(tokenize['input_ids'])
  train_data['token_type_ids'].append(tokenize['token_type_ids'])
  train_data['attention_mask'].append(tokenize['attention_mask'])
  train_data['labels'].append(raw_datasets['train']['label'][i])


valid_data = {'input_ids': [],
              'token_type_ids': [],
              'attention_mask': [],
              'labels': []}

for i in range(len(raw_datasets['validation'])):
  tokenize = tokenizer(raw_datasets['validation']['sentence1'][i], raw_datasets['validation']['sentence2'][i], truncation=True)
  valid_data['input_ids'].append(tokenize['input_ids'])
  valid_data['token_type_ids'].append(tokenize['token_type_ids'])
  valid_data['attention_mask'].append(tokenize['attention_mask'])
  valid_data['labels'].append(raw_datasets['validation']['label'][i])

from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self, data):  
        self.data = data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, index):
        return {'input_ids': self.data['input_ids'][index],
                'token_type_ids': self.data['token_type_ids'][index],
                'attention_mask': self.data['attention_mask'][index],
                'labels': self.data['labels'][index]}

train_dataset = TestDataset(train_data)
eval_dataset = TestDataset(valid_data)

"""
수정부분 종료
"""

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)


#모델 불러오기
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

#Optimizer 설정
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# 스케줄러 설정
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

#GPU사용
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#tqdm으로 진행사항 확인
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

#학습시작
model.train()
for epoch in range(num_epochs):
  for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

"""
수정부분 
 - 정확도와 f1_score를 sklearn을 사용해서 표현
"""
#정확도 및 f1값 저장
from sklearn.metrics import accuracy_score, f1_score
accuracy = 0
f1 = 0

#평가모드
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()} #gpu 사용을 위한 코드 model이 gpu를 사용하기 때문에 데이터도 gpu 로
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    #모델이 GPU에서 작동하고 데이터가 GPU에 있기 때문에 CPU로 옮기는 과정이 필요함
    acc = accuracy_score(batch['labels'].to(torch.device('cpu')), predictions.to(torch.device('cpu')))
    accuracy += acc
    f = f1_score(batch['labels'].to(torch.device('cpu')), predictions.to(torch.device('cpu')))
    f1 += f

accuracy /= len(eval_dataloader)
f1 /= len(eval_dataloader)

#정확도와 f1출력
print("Accuracy: ", accuracy, "\n", "F1_Score: ", f1)