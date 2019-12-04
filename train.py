import spacy
import random
import json

raw_data = []
raw_data_count = 0
with open('data-dev.txt', encoding='utf-8') as f:
    for line in f:
        if raw_data_count == 20000:
            break
        raw_data_count = raw_data_count + 1
        print(raw_data_count)
        raw_data.append(json.loads(line))
print("Raw data : " + str(raw_data_count))

TRAIN_DATA = []
TRAIN_DATA_count = 0
for rd in raw_data:
    find = str(rd['address']).strip().lower().find(str(rd['street']).strip().lower())
    if find != -1:
        TRAIN_DATA_count = TRAIN_DATA_count + 1
        TRAIN_DATA.append((str(rd['address']).strip().lower(),
                           {'entities': [(find, find + len(str(rd['street']).strip()), 'Street')]}))
print("Train data : " + str(TRAIN_DATA_count))


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training(device=0)
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

modelfile = 'auto_split_street_address_model2'
prdnlp.to_disk(modelfile)
