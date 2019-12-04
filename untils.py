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
print(spliter.run('Shop 07A/GF, TT thương mại Big C Huế, 174 Bà Triệu, Phú Hội, TP Huế'))
