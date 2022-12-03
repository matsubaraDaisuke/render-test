import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


from PIL import Image # $ pip install pillow
from google_drive_downloader import GoogleDriveDownloader as gdd


classes = ["男","女"]
image_size = 50

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#model = load_model('./model.h5')#学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == None:
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print("test")
            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath,target_size=(image_size,image_size))
            #print(img)
            img = image.img_to_array(img)
            data = np.array([img])
            print(data.shape)
            # #変換したデータをモデルに渡して予測する
            #result = model.predict(img)[0]

            gdd.download_file_from_google_drive(file_id='1V0XSyCRYZ4w9vqI89mthqJZhSdO8KJtr',
                                    dest_path='./data/model.h5',
                                    unzip=False)
            model = load_model("./data/model.h5")
            result = model.predict(data)
            del model
            del file
            os.remove(os.path.join(UPLOAD_FOLDER, filename))
            os.remove("./data/model.h5")

            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    app.run(port=3000)
    #app.run(debug=True)