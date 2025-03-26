from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import BadRequest
import re
import joblib
import numpy as np
import os
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import tensorflow as tf
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sqlite3

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for image updates

# Connect to the SQLite database
conn = sqlite3.connect('instance/predictions.db', check_same_thread=False)
cur = conn.cursor()

def generate_graph(disease):
    query = "SELECT age FROM prediction WHERE disease = ?"
    cur.execute(query, (disease,))
    ages = [age[0] for age in cur.fetchall()]

    plt.figure(figsize=(10, 5))
    plt.scatter(ages, [disease] * len(ages), alpha=0.5)  # Adjust the plot type if necessary
    plt.title(f'Age Distribution for {disease}')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    image_path = f'static/{disease}_plot.png'
    plt.savefig(image_path)
    plt.close()
    return image_path

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    image_path = None
    if request.method == 'POST':
        selected_disease = request.form.get('disease')
        if selected_disease:
            image_path = generate_graph(selected_disease)

    # Fetch all distinct diseases for dropdown
    cur.execute("SELECT DISTINCT disease FROM prediction")
    diseases = [row[0] for row in cur.fetchall()]
    return render_template('dashboard.html', diseases=diseases, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
