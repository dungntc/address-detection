import spacy
import json

nlp = spacy.load('first')

# test_text = input("Enter your testing text: ")
# doc = nlp(test_text)
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)


raw_data = []
raw_data_count = 0
with open('data-test.txt', encoding='utf-8') as f:
    for line in f:
        if raw_data_count == 1000:
            break
        raw_data.append(json.loads(line))
        raw_data_count = raw_data_count + 1

find_data_count = 0
right_data_count = 0
for data in raw_data:
    address = str(data['address']).strip()
    street = str(data['street']).strip()
    doc = nlp(address.lower())
    if len(doc.ents) > 0:
        data['new_street'] = address[doc.ents[0].start_char: doc.ents[0].end_char]
        find_data_count = find_data_count + 1
        if doc.ents[0].text == street.lower():
            right_data_count = right_data_count + 1
    else:
        data['new_street'] = ''

print(raw_data_count, find_data_count, right_data_count)
print(float(right_data_count / raw_data_count * 100))