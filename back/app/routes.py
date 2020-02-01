# -*- coding: utf-8 -*-
from app import app
from flask import render_template
from app.forms import DataSendingForm


@app.route('/')
@app.route('/data')
def data():
    form = DataSendingForm()
    return render_template('index.html', title = 'Hm...', form = form)
