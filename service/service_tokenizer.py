from sanic import Sanic
from sanic.response import text,json,html

from text2vec.tokenizer import Tokenizer
tokenizer_service= Sanic("tokenizer")

class TokenizerSerive(object):
    def __init__(self):
        self.name = "tokenize serive"
        self.tokenize = Tokenizer()

    def __str__(self):
        return self.name

    def tokenize_(self,sentences,tokenize_label):
        return self.tokenize.tokenize(sentences,tokenize_label)



@tokenizer_service.route("/intelligencAlgor/tokenizer/tokenize",methods = ['POST','GET'])
async def tokenizer_serive(request):
    user_send = []
    user_send = request.args
    sentences = user_send['sentences']
    tokenize_label = user_send['tokenize_label']
    tokenizerserive = TokenizerSerive()
    out_result = tokenizerserive.tokenize_(sentences[0],tokenize_label[0])
    return json({"tokenizer_result":str(out_result)})






