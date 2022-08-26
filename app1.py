from cgitb import handler
from email import message
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import session as login_session
from flask import make_response
import pandas as pd
import os
from face_verification.new import facerecog, face_match
from Biometric.biometric import biometrics

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            return redirect(url_for('select'))
    return render_template('login.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('index1.html')

@app.route('/biometric', methods=['GET', 'POST'])
def biometric():
    if request.method == 'POST':
        img = request.files['image']
        img.save(os.path.join('static/frames', img.filename))

        # If img.jpg already exists, delete it
        if os.path.exists('static/frames/img.tif'):
            os.remove('static/frames/img.tif')
        # Rename the image to image.jpg
        os.rename(os.path.join('static/frames', img.filename), os.path.join('static/frames', 'img.tif'))

        return redirect(url_for('searching',messages=0))
    return render_template('biometric.html')

@app.route('/aadhar', methods=['GET', 'POST'])
def aadhar():
    if request.method == 'POST':
        img = request.files['image']
        img.save(os.path.join('static/frames', img.filename))

        # If img.jpg already exists, delete it
        if os.path.exists('static/frames/img.jpg'):
            os.remove('static/frames/img.jpg')
        # Rename the image to image.jpg
        os.rename(os.path.join('static/frames', img.filename), os.path.join('static/frames', 'img.jpg'))
        
        return redirect(url_for('searching',messages=1))

    return render_template('aadhar.html')

@app.route('/searching', methods=['GET', 'POST'])
def searching():
    mess = request.args['messages']
    print(mess)
    persondata = ''

    try:
        if int(mess)==1:
                result = face_match('static/frames/img.jpg','face_verification/data2.pt')
                
                if result=="none":
                    file='none'
                else:
                    folder = os.path.join('static/photos',result[0])
                    print("Photo matched with: ",result[0])
                    file = os.path.join(folder,os.listdir(os.path.join('static/photos',result[0]))[0])
                    namee = result[0]

        elif int(mess)==2:
            result = facerecog()

            if result=="none":
                file='none'
            else:
                folder = os.path.join('static/photos',result[0])
                print("Face matched with: ",result[0])
                file = os.path.join(folder,os.listdir(os.path.join('static/photos',result[0]))[0])
                namee = result[0]
    
        else:
            result = biometrics('static/frames/img.tif')

            if result=="none":
                file='none'
                os.remove('static/frames/img.tif')
            else:
                folder = os.path.join('static/photos',result)
                print("Fingerprint matched with: ",result)
                os.remove('static/frames/img.tif')
                file = os.path.join(folder,os.listdir(os.path.join('static/photos',result))[0])
                namee = result
    
    except:
        file = 'none2'

    data = pd.read_csv("static/data/data.csv")
    if 'none' not in file:
        nameset = data['Name']

        for i in range(len(nameset)):
            if namee.lower() in nameset[i].lower():
                print("Row number",i+1)
                persondata = data.iloc[i,:]
                break

        print(persondata)
        print(persondata[0])

    return render_template('searching.html',img_path=file, persondat=zip(data.columns,persondata))

@app.route('/cctv', methods=['GET', 'POST'])
def cctv():
    if request.method=='POST':
        return redirect(url_for('searching',messages=2))
    
    return render_template('cctv.html')


if __name__ == '__main__':
    app.run(debug=True)

