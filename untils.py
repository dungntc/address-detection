import spacy
import os

class AutoSplitStreetAddress:
    def __init__(self,model_url):

        self.nlp = spacy.load(model_url)

    def run(self, address):
        address = str(address).strip()
        doc = self.nlp(address.lower())
        if len(doc.ents) > 0:
            return address[doc.ents[0].start_char: doc.ents[0].end_char]
        else:
            return ''


# sample
spliter = AutoSplitStreetAddress(os.getcwd()+'/auto_split_street_address_model')
print('-----------------------------')
print('\n')
print('Ví dụ : ')
print('Địa chỉ chi tiết : Shop 07A/GF, TT thương mại Big C Huế, 174 Bà Triệu, Phú Hội, TP Huế')
print('Đường/phố : '+spliter.run('Shop 07A/GF, TT thương mại Big C Huế, 174 Bà Triệu, Phú Hội, TP Huế'))
print('\n')
print('-----------------------------')
print('\n')
print('Thử với ví dụ khác : ')
while True:
    sample_address = input('Địa chỉ chi tiết : ')
    if sample_address=='end':
        break
    print('Đường phố : '+spliter.run(sample_address)+'\n')
