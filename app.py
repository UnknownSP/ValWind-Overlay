import time
from subprocess import Popen

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit


app = Flask(__name__)
app.secret_key = '&(*(**((*@@@#$333(*(*221'
socket_io = SocketIO(app, cors_allowed_origins="*")

app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

cors = CORS(app)

@app.route('/start_match', methods=['GET', 'POST'])
@cross_origin(allow_headers=['*'])
def start_match():
    Popen('python test.py')
    
@socket_io.on('test')
def new_event(data):
    global match_details
    print("test", data)
    
    emit('test',  {
         'match_details': data}, broadcast=True, include_self=False)

if __name__ == "__main__":
    socket_io.run(app, port=4445, host="127.0.0.1", debug='true')
    ## start_frame_grabbing()
    #sio = socketio.Server(cors_allowed_origins='*') # CORSのエラーを無視する設定
    ##sio.register_namespace(MyCustomNamespace('/test')) # 名前空間を設定
    #app = socketio.WSGIApp(sio) # wsgiサーバーミドルウェア生成
    #eventlet.wsgi.server(eventlet.listen(('127.0.0.1',4445)), app) # wsgiサーバー起動