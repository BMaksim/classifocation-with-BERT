import sys
sys.path.insert(0, 'D:\Projects\classifocation-with-BERT/test')

# -*- coding: utf-8 -*-
from app import app
from flask import render_template
from app.forms import DataSendingForm
from flask import render_template, redirect
from fsomeFile import get_answer


@app.route('/')
@app.route('/answer/<answer>')
def answer(answer):
    return render_template('answer.html', answer = answer)
@app.route('/data', methods=['GET', 'POST'])
def data():
    form = DataSendingForm()
    if form.validate_on_submit():
        form = DataSendingForm()
        data = form.data.data
        answer = get_answer(data)
        return redirect('/answer/{}'.format(answer))
    return render_template('index.html', title = 'Hm...', form = form)
