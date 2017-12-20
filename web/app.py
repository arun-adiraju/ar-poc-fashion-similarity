import shutil

import numpy as np
from flask import Flask, render_template, request, json
from flask.json import JSONEncoder
from service import get_label, get_similar_images
from similaritySearch import ImageData
from similaritySearch import get_label_features_mapping1
from werkzeug import secure_filename

app = Flask(__name__)

uploadedImageFolder = "static/uploadedImages/"


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/testupload')
def test_upload_file():
    return render_template('test.html')


@app.route('/uploader', methods=['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['file']

        f.save(secure_filename(f.filename))
        shutil.move(f.filename, uploadedImageFolder)
        return 'file uploaded successfully'


@app.route('/templatetest')
def template_test():
    return render_template('similarImagesResults.html', my_string="img.jpg!", my_list=['img.jpg'])


@app.route('/find_similar', methods=['GET', 'POST'])
def find_similar_images():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        shutil.move(f.filename, uploadedImageFolder + f.filename)
        similar_images, label_name = get_label(f.filename)
        return render_template('similarImagesResults.html', input_label=label_name, similar_list=similar_images,
                               input_file=f.filename)


@app.route('/initialize', methods=['GET'])
def initialize_app():
    if request.method == 'GET':
        # session['feature_mappings'] = get_all_feature_vectors()
        # print(session['feature_mappings'])
        return 'initialized'


@app.route('/find_similar_with_session', methods=['GET', 'POST'])
def find_similar_images_with_session():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        shutil.move(f.filename, uploadedImageFolder + f.filename)
        similar_images, label_name = get_similar_images(f.filename)
    return render_template('similarImagesResults.html', input_label=label_name, similar_list=similar_images,
                           input_file=f.filename)

    # if 'feature_mappings' in session:
    #     # print(session['feature_mappings'])
    #     # label_features_mapping = session['feature_mappings']
    #     # print(label_features_mapping)
    #     similar_images, label_name = get_similar_images_using_session(f.filename)
    #     return render_template('similarImagesResults.html', input_label=label_name, similar_list=similar_images, input_file=f.filename)
    # else:
    #     print('session not initialized')
    #     return 'application not booted'


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ImageData):
            # Implement code to convert Passport object to a dict
            return json.dumps(obj.__dict__)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        else:
            JSONEncoder.default(self, obj)


if __name__ == '__main__':
    get_label_features_mapping1()
    app.secret_key = 'super secret key'
    SESSION_TYPE = 'filesystem'
    app.config['SESSION_PERMANENT'] = True
    app.json_encoder = CustomJSONEncoder

    app.run(debug=False)
