# disease-api-server
This repo only showing serving scripts.


## step 1
Image files uploaded from user PC. and this image is `base64` encoded.

`predict` method return ndarray predicted bounding box marked.
~~~
class SurvedModel:

    def __init__(self):
        '''
        Model should be loaded on memory here.  
        '''
        # self.your_model = ~~

    def predict (self, img):
        '''
        Preprocessing & inference & postprocessing part.
        # img;attribute = {shape:[H, W, 3],  type : ndarray}
        # return;attribute = {shape : [H, W, 3], type : ndarray}

        # return your_postprocessing(self.your_model(your_preprocessing(img)))
        # Draw box on the imag
~~~

### step 3
Pack inferences data as `json` and return.
~~~
 # map as json.
    output_json = json.dumps({'data': returns_dat.decode(),
                              'info': {'height': output.shape[0], 'width': output.shape[1], 'channel': output.shape[2]},
                              'time': f'{(str(time.time() - time_start))[:5]}s',
                              'is_gpu': tf.test.is_gpu_available()})
~~~


# Front side
Upload image as `base64` encoded.   
After inference, get encoded image data and decode.
~~~
# Set content_type to header.
content_type = 'application/json'
headers = {'content-type': content_type}

# upload image string array data.
img_file = request.files['file'].stream.read()

img = cv2.imdecode(np.fromstring(img_file, np.uint8), cv2.IMREAD_COLOR)
print(img)

# map to json.
send = base64.b64encode(np.array(img))

request_json = json.dumps({'input_img': send.decode(),
                            'info': {
                                'height': img.shape[0],
                                'width': img.shape[1],
                                'channel': img.shape[2]
                            }
                            })

print('request_json\n', request_json)

# http request.
response = requests.post('<address>', data=request_json, headers=headers)
# print(response)

# ['data', 'time', 'is_gpu']
response_json = response.json()

# change to numpy array.
r = base64.decodebytes(response_json['data'].encode())
response_dat = np.fromstring(r, dtype=np.float)
print(response_dat)

response_dat = response_dat.reshape((response_json['info']['height'],
                                        response_json['info']['width'],
                                        response_json['info']['channel']))

# decodeed numpy image.
print(response_dat)
plt.figure(figsize=(7, 7))
plt.imshow(response_dat)
timenow = str(time.time())
fname = os.path.join('<path>', timenow + '.png')
plt.axis('off')
plt.savefig(fname)

~~~