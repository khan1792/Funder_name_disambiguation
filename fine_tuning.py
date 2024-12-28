import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
import random



logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


# load and merge data
pos = pd.read_csv('pos.csv')
# set the label as 1 (positive)
pos['pos'] = 1
# the data after this step is [name 1, name 2, label=1, data_source]

# load negative pairs
neg_wos = pd.read_csv('negative_wos.csv')
neg_wos = neg_wos[['match_name', 'primary', 'label']]
neg_wos.columns = ['wos', 'name', 'label']
neg_wos['source'] = 'wos'
neg_ror = pd.read_csv('negative_ROR.csv')
neg_ror = neg_ror[['match_name', 'primary', 'label']]
neg_ror.columns = ['wos', 'name', 'label']
neg_ror['source'] = 'ror'
neg_ror = neg_ror.sample(n = 500000, random_state = 1) # randomly get 50,0000 negative ror instances

# merge ror and wos for negative instances
neg = pd.concat([neg_wos, neg_ror], ignore_index = True)
# set the label as 0
neg['pos'] = 0
# the data after this step is [name 1, name 2, label=0, data_source]
# name 1 is "wos" (anchor name), name 2 is "name" in terms of real column name in my data. 

# merge positive and negative instances and shuffle it
ddd = pd.concat([pos, neg], ignore_index = True).sample(frac = 1, random_state = 1, ignore_index = True)
# the data after this step is [name 1, name 2, label, data_source]

# split training and validation dataset
train = ddd[40000:]
train.index = range(len(train))
test = ddd[:40000]
test.index = range(len(test))


    
# load pre-trained model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# change it to sentence transformer format
train_examples_rank = []
train_examples_contrast = []
train_batch_size = 220

# contrastive training data
for i in range(len(train)):
    train_examples_contrast.append(InputExample(texts=[train['wos'][i], train['name'][i]], label = train['label'][i]))
    
train_dataloader_ConstrativeLoss = DataLoader(train_examples_contrast, shuffle=True, batch_size=train_batch_size)

# multiple ranking training data
train1 = train[train['label']==1]
train1.index = range(len(train1))
train_examples_rank = []
for i in range(len(train1)):
    train_examples_rank.append(InputExample(texts=[train1['wos'][i], train1['name'][i]], label = train1['label'][i]))    
    train_examples_rank.append(InputExample(texts=[train1['name'][i], train1['wos'][i]], label = train1['label'][i])) 
    


train_dataloader_MultipleNegativesRankingLoss = DataLoader(
    train_examples_rank, shuffle=True, batch_size=train_batch_size
)

# the distance is coscine similarity distance
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE


# set loss functions
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model=model)
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=0.5)


# set evaluators
# I used three evaluators, but not all of them are mandatory. You may just use Classification and Information Retrieval evaluators
evaluators = []

###### Classification ######
# Given (name 1, name 2), do they refer to the same funder or different funder?
# The evaluator will compute the embeddings for both names and then compute
# a cosine similarity. If the similarity is above a threshold, we are positive pairs.


binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(test['wos'].tolist(), test['name'].tolist(), test['label'].tolist())

#add the classification evaluator
evaluators.append(binary_acc_evaluator)



###### Same funder Mining ###### (this evaluator is optional)
# Given a large corpus of funder names, identify all names for the same funder

# create a evaluation data for same funder mining
test1 = test[test['wos']!=test['name']][test['pos']==1]
test1 = test1.drop_duplicates(subset = ['wos']).drop_duplicates(subset = ['name'], ignore_index = True)

test_values = test1['wos'].tolist() + test1['name'].tolist()
test_keys = range(len(test1)*2)
test_keys = [str(x) for x in test_keys]

res_pm = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
test1['id1'] = range(len(test1))
test1['id1'] = test1['id1'].apply(lambda x: str(x))
test1['id2'] = range(len(test1), len(test1)*2)
test1['id2'] = test1['id2'].apply(lambda x: str(x))
sen = list(map(list, zip(test1['id1'].tolist(), test1['id2'].tolist())))
paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(res_pm, sen, name="dev")


# add the mining evaluator
evaluators.append(paraphrase_mining_evaluator)
# you may skip the whole mining evaluator section if you dont want to use this evalautor)


###### Information Retrieval ######
# Given a funder name and a large number of funder names, find the most relevant funder
# create a dataset for evalaution
# data format - "0" the number is the index. name_set is a set of names that refer to the the same funder. lll is the length of a name set (how many names are in a the name set)

# {
#   name_set: {"0": [name 01, name 02, name 03],
#              "1": [name 11, name 12, name 13, name 14]
#              "..." : [...]}
#   lll:      {"0": 3,
#              "1": 4
#              "..." : ...}}

lrt = pd.read_json('lrt_data.json')


# create a evaluation data based on the above data. 
n = 0
corpus_res = {}
corp = []
for i in lrt['name_set'].index:
    test_keys = list(range(n, n+lrt['lll'][i]))
    test_keys = [str(x) for x in test_keys]
    test_values = list(lrt['name_set'][i])
    t = set(test_keys)
    n = n+lrt['lll'][i]
    res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
    corp.append(t)
    corpus_res.update(res)
    
lrt['name_keys'] = corp  
test1['final_qr'] = range(len(corpus_res), len(corpus_res) + len(test1))
test1['final_qr'] = test1['final_qr'].apply(lambda x: str(x))

test_keys = test1['final_qr']
test_values = test1['wos'].tolist()
qr_res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}

final = pd.merge(test1, lrt, how="inner", on=["id"])

test_keys = final['final_qr']
test_values = final['name_keys']
rele_res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}

ir_evaluator = evaluation.InformationRetrievalEvaluator(qr_res, corpus_res, rele_res)


#add the retrieval evaluator
evaluators.append(ir_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])



# fine-tune
model_save_path = "multi_task_online" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)



model.fit(
    train_objectives=[
        (train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss),
        (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss),
    ],
    evaluator=seq_evaluator,
    epochs=10,
    warmup_steps=2000,
    evaluation_steps = 180,
    output_path=model_save_path,
    show_progress_bar = True)