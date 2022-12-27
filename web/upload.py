from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import os
from dlmodel import *
from PIL import Image
import io
#########
# model
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import os
# import matplotlib.pyplot as plt
################################
# sql
import sqlite3 as sql
conn = sql.connect('database.db')
print("Opened database successfully")

conn.execute(
    'CREATE TABLE IF NOT EXISTS students (pet TEXT, info TEXT)')
# conn.execute(
#     'CREATE TABLE IF NOT EXISTS students (name TEXT, addr TEXT, city TEXT, pin TEXT)')
print("Table created successfully")
conn.close()
########################
#image
import pandas as pd


app = Flask(__name__)

# @app.route('/index')
# def index():
#     return render_template('index.html')

#HTML 렌더링
@app.route('/')
def home_page():
	return render_template('index.html')

# # 모델 
# @app.route('/model', methods = ['GET', "POST"])
# def model():
    

# 파일 리스트
@app.route('/list')
def list_page():
	file_list = os.listdir('C:/dev/vscod_workspace/template1/static/uploads')
	html = """<center><a href="/">홈페이지</a><br><br>""" 
	html += "file_list: {}".format(file_list) + "</center>"
	return html

#업로드 HTML 렌더링
@app.route('/upload')
def upload_page():
	return render_template('index.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		#저장할 경로 + 파일명
		f.save('C:\\dev\\vscod_workspace\\template1\\uploads\\' + secure_filename(f.filename))
		#flash("사진 업로드 성공!")
		#msg = "error in insert operation"
		


		return render_template('index.html')
# booking
@app.route('/booking')
def booking():
    return render_template('booking.html')

# price
## list
@app.route('/price')
# def price():
#     con = sql.connect("database.db")
#     con.row_factory = sql.Row

#     cur = con.cursor()
#     cur.execute("select * from pet")

#     rows = cur.fetchall()
#     return render_template("price.html", rows=rows)
def price():
    pet_info = pd.read_excel('pet.xlsx')
    rows = pet_info.columns
    return render_template("price.html", rows=rows)
    # breed = breed, basic=basic....
    
    
# 모델    
@app.route("/predict", methods=["POST"])
def preds():
	data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
	if request.method == "POST":
		pic = request.files['file']
		image = request.files['file'].read()
		image = Image.open(io.BytesIO(image))
		image = prepare_image(image, target=(480, 480))
		image = create_data_batches(image)
		preds = loaded_full_model.predict(image) 
		custom_pred_labels = [get_pred_label(preds)]
  
	# 견종이름 추출
	pet_info = pd.read_excel('pet.xlsx')
	pet = pet_info[pet_info['Breed'] == custom_pred_labels[0]]

	Breed = pet.iloc[0][0]
	basic = int(pet.iloc[0][1])
	plus = int(pet.iloc[0][2])
	primium = int(pet.iloc[0][3])
	feature1 = pet.iloc[0][4]
	feature2 = pet.iloc[0][5]
	feature3 = pet.iloc[0][6]
	feature4 = pet.iloc[0][7]
	if Breed == 'flat-coated_retriever':
		Breed = '포메라니안'
	elif Breed == 'norwegian_elkhound':
		Breed = '미니어쳐 푸들'
	elif Breed == 'ibizan_hound':
		Breed = '토이 푸들'

    

	
   
		
      
      
    #return render_template("price.html", rows=rows)

	return render_template('price.html', pred=custom_pred_labels, pic=pic, 
                        Breed=Breed,basic = basic, plus = plus, primium = primium,
                        feature1 = feature1, feature2 = feature2, feature3=feature3, feature4=feature4)






    

#서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)
    
    
    



