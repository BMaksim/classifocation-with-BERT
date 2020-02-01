# -*- coding: utf-8 -*-
from app import app
from flask import render_template
from app.forms import DataSendingForm
from flask import render_template, flash, redirect


@app.route('/')
@app.route('/data', methods=['GET', 'POST'])
def data():
    form = DataSendingForm()
    if form.validate_on_submit():
        flash('Just a second')
        data = form.data.data
        answer = get_answer(data)
        return render_template('answer.html', answer = answer)
    return render_template('index.html', title = 'Hm...', form = form)
