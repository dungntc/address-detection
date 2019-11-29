import spacy
import random
import json


# raw_data = [{'address': 'Số 82 Kim Mã, P. Kim Mã, Q. Ba Đình, TP. Hà Nội', 'street': 'Kim Mã'},
#             {'address': '334 Hải Phòng, P. Chính Gián, Q. Thanh Khê, TP. Đà Nẵng', 'street': 'Hải Phòng'},
#             {'address': '74 Bạch Đằng, P. Hải Châu 1, Q. Hải Châu, TP. Đà Nẵng', 'street': 'Bạch Đằng'},
#             {'address': '221/5 Lê Văn Sỹ, P.13, Q.3, TP.Hồ Chí Minh', 'street': 'Lê Văn Sỹ'},
#             {'address': 'Tầng 5, Trung tâm thương mại MAC PLAZA, số 10 Trần Phú, P. Mộ Lao, Q. Hà Đông, TP Hà Nội',
#              'street': 'Trần Phú'},
#             {'address': '124 Bạch Mai, P. Cầu Dền, Q. Hai Bà Trưng, TP. Hà Nội', 'street': 'Bạch Mai'},
#             {'address': 'Số 43 Nguyễn Trường Tộ, P. Nguyễn Trung Trực, Q. Ba Đình, TP. Hà Nội',
#              'street': 'Nguyễn Trường Tộ'},
#             {'address': 'Số 185-187 Nguyễn Thái Học,  P. Phạm Ngũ Lão, Q. 1, TP. Hồ Chí Minh',
#              'street': 'Nguyễn Thái Học'},
#             {'address': 'Số 27-29-31 Lý Tự Trọng, P. An Phú, Q. Ninh Kiều, TP. Cần Thơ, Việt Nam',
#              'street': 'Lý Tự Trọng'},
#             {'address': '29 Trần Quốc Hoàn, P. Dịch Vọng Hậu, Q. Cầu Giấy, TP. Hà Nội', 'street': 'Trần Quốc Hoàn'},
#             {'address': 'Lầu 1 - Lầu 2, số 22 - 22 Bis Lê Thánh Tôn, P. Bến Nghé, Q. 1, TP.Hồ Chí Minh, Việt Nam',
#              'street': 'Lê Thánh Tôn'},
#             {'address': 'Crescent Mall, 101 Tôn Dật Tiên, Phường Tân Phong, Quận 7, Thành phố Hồ Chí Minh',
#              'street': 'Tôn Dật Tiên'},
#             {'address': 'X45, đường số 17, KDC Ehome 4 Nam Long, Phường Vĩnh Phú, Thị xã Thuận An, T. Bình Dương',
#              'street': 'Số 17'},
#             {'address': 'Sn 306, Hồng Hà, Tổ 12, P. Cốc Lếu, TP. Lào Cai, T. Lào Cai', 'street': 'Hồng Hà'},
#             {'address': 'vệ linh, xã phù linh, H. Sóc Sơn, TP. Hà Nội', 'street': 'Vệ Linh'},
#             {'address': 'Số 44, ngách 203/2, ngõ 255, phố Thanh Am, P. Thượng Thanh, Q. Long Biên, TP. Hà Nội',
#              'street': 'Thanh Am'},
#             {'address': '54 Lý Quốc Sư, P. Hàng Trống, Q. Hoàn Kiếm, TP. Hà Nội', 'street': 'Lý Quốc Sư'},
#             {'address': 'Số 112D4 ngõ 4C Đặng Văn Ngữ, P. Trung Phụng, Q. Đống Đa, TP. Hà Nội',
#              'street': 'Đặng Văn Ngữ'},
#             {'address': '283 Phan Châu Trinh, P. Phước Ninh, Q. Hải Châu, TP. Đà Nẵng', 'street': 'Phan Châu Trinh'},
#             {'address': 'Tổ dân phố Tân Tây Đô 1, xã Tân Lập, huyện Đan Phượng, thành phố Hà Nội',
#              'street': 'Tân Tây Đô 1'}]

raw_data = []
raw_data_count = 0
with open('data-dev.txt',encoding='utf-8') as f:
    for line in f:
        if raw_data_count == 10000:
            break
        raw_data.append(json.loads(line))
        raw_data_count = raw_data_count + 1
print(raw_data_count)

TRAIN_DATA = []
TRAIN_DATA_count = 0
for rd in raw_data:
    find = str(rd['address']).strip().lower().find(str(rd['street']).strip().lower())
    if find != -1:
        TRAIN_DATA_count = TRAIN_DATA_count + 1
        TRAIN_DATA.append((str(rd['address']).strip().lower(),
                   {'entities': [(find, find + len(str(rd['street']).strip()), 'Street')]}))
    # else:
        # print(rd['address'], rd['street'])
print(TRAIN_DATA_count)


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
        optimizer = nlp.begin_training()
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


prdnlp = train_spacy(TRAIN_DATA, 3)

# Save our trained Model
modelfile = 'first'
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
test_text = test_text.lower()
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)