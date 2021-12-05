from flask import Flask, Response, request
from inference.inference import SurvedModel
import time
import json
import tensorflow as tf
import numpy as np
import base64

app = Flask(__name__)
model = SurvedModel()


@app.route('/get_paprika_result', methods=['POST'])
def predict():
    ############################################### Test code #################################################
    # prev = time.time()
    #
    # # Load Image
    # img = cv2.imread('/opt/project/paprika_model/test.jpg')
    # output = model.predict(img)
    #
    # return f'Connected : {output.shape}, Time : {time.time() - prev}s , GPU : {tf.test.is_gpu_available()}'
    ###########################################################################################################

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

    # predict.
    output = model.predict(input_img)
    print('CHANGE +++++++++++++++++++++++++++++\n', output.shape)
    print(output)

    returns_dat = base64.b64encode(np.array(output))

    # map as json.
    output_json = json.dumps({'data': returns_dat.decode(),
                              'info': {'height': output.shape[0], 'width': output.shape[1], 'channel': output.shape[2]},
                              'time': f'{(str(time.time() - time_start))[:5]}s',
                              'is_gpu': tf.test.is_gpu_available()})

    # dat = base64.b64decode(returns_dat.decode().encode())
    # dat = np.fromstring(dat, dtype=np.float)
    # print(dat.shape)

    # response.
    return Response(response=output_json, status=200, mimetype="application/json")


# # 파일 업로드 처리
# @app.route('/fileUpload', methods=['POST'])
# def upload_file():
#     # Set content_type to header.
#     content_type = 'application/json'
#     headers = {'content-type': content_type}
#
#     # upload image string array data.
#     img_file = request.files['file']
#
#     # dat string.
#     request_data = img_file.read()
#
#     # map to json.
#     request_json = jsonpickle.encode({'input_img': request_data})
#
#     # http request.
#     response = requests.post(test_url, data=request_json, headers=headers)
#
#     # ['data', 'time', 'is_gpu']
#     response_json = json.loads(response.text)
#
#     # change to numpy array.
#     response_dat = np.fromstring(response_json['data'], np.uint8)
#
#     # decodeed numpy image.
#     output_final = cv2.imdecode(response_dat, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50001)
