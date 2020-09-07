#  Created at 2019/4/10                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com
import web
import json
from model_api.model_pipeline import model_pipeline
pipeline=model_pipeline()
urls = (
    '/', 'index',
    '/model','model',

)

class MyApplication(web.application):
 def run(self, port=4444, *middleware):
    func = self.wsgifunc(*middleware)
    return web.httpserver.runsimple(func, ('192.168.10.27', port))

app = MyApplication(urls,globals())
class index:
    def GET(self):
        return "Hello, world!"

class model:
    def GET(selfs):
        return "model"
    def POST(self):
        js_obj = json.loads(web.data())
        obj = js_obj['obj']
        weathers = js_obj['weathers']
        pipeline(obj)
        data={}
        data["result"]=pipeline.model_predict(weathers=weathers).tolist()
        return data



