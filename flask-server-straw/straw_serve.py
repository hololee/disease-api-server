from flask import Flask, Response, request
from strawbery import SurvedModel
import cv2
import time
import json
import torch
import numpy as np
import base64
from imageio import imwrite

app = Flask(__name__)
model = SurvedModel()


@app.route('/get_straw_result', methods=['POST'])
def predict():
    # For check inference time.
    time_start = time.time()

    # Get json from client. request of client should be mimetype="application/json"
    upload_json = request.json
    '''
    upload_json = {input_img : <DAT>}
    '''
    print(upload_json['input_img'])

    r = base64.decodebytes(upload_json['input_img'].encode())
    input_img = np.fromstring(r, dtype=np.uint8)

    input_img = input_img.reshape((upload_json['info']['height'], upload_json['info']['width'], upload_json['info']['channel']))
    print('CHANGE +++++++++++++++++++++++++++++\n', input_img.shape)

    # resize imag.
    input_img = cv2.resize(input_img, dsize=(448, 448), interpolation=cv2.INTER_CUBIC)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # predict.
    output = model.predict(input_img) / 255.0

    print('CHANGE +++++++++++++++++++++++++++++\n', output.shape)

    returns_dat = base64.b64encode(np.array(output))
    print(output)
    imwrite('./test.jpg', output)

    # map as json.
    output_json = json.dumps({'data': returns_dat.decode(),
                              'info': {'height': output.shape[0], 'width': output.shape[1], 'channel': output.shape[2]},
                              'time': f'{(str(time.time() - time_start))[:5]}s',
                              'is_gpu': torch.cuda.is_available()})


    return Response(response=output_json, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50002)
