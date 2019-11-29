import spacy
import json

nlp = spacy.load('first')

# test_text = input("Enter your testing text: ")
# doc = nlp(test_text)
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)


raw_data = []
raw_data_count = 0
with open('data-dev.txt', encoding='utf-8') as f:
    for line in f:
        if raw_data_count == 15000:
            break
        raw_data.append(json.loads(line))
        raw_data_count = raw_data_count + 1

find_data_count = 0
right_data_count = 0
for data in raw_data:
    doc = nlp(str(data['address']).strip().lower())
    if len(doc.ents) > 0 is not None:
        find_data_count = find_data_count + 1
        if doc.ents[0].text == str(data['street']).strip().lower():
            right_data_count = right_data_count + 1
            print('true: ' + doc.ents[0].text + '-----------res: ' + str(
                data['street']).strip().lower() + '-------------add: ' + str(data['address']))
        else:
            print('false: ' + doc.ents[0].text + '-----------res: ' + str(
                data['street']).strip().lower() + '-------------add: ' + str(data['address']))

print(raw_data_count, find_data_count, right_data_count)
print(float(right_data_count / raw_data_count * 100))
