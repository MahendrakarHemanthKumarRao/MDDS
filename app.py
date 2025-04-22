import base64
import logging
import os
import random
import re
import string
from io import BytesIO
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from matplotlib.figure import Figure
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from threading import Thread
from dashboard import generate_graph
import pandas as pd
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///MDDSA09.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD'] = 'dbimages'

os.makedirs(app.config['UPLOAD'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD'], exist_ok=True)

# Models
class Prediction(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    disease = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class User(db.Model, UserMixin):
    __tablename__ = 'loginuser'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(15), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
'''
class Submitdata(db.Model):
    __tablename__ = 'submitdata'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(10), unique=True, nullable=False)
    username = db.Column(db.String(100))
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    # Liver-related fields
    total_bilirubin = db.Column(db.Float)
    direct_bilirubin = db.Column(db.Float)
    alkaline_phosphotase = db.Column(db.Float)
    alamine_aminotransferase = db.Column(db.Float)
    aspartate_aminotransferase = db.Column(db.Float)
    total_proteins = db.Column(db.Float)
    albumin = db.Column(db.Float)
    albumin_globulin_ratio = db.Column(db.Float)
    # Kidney-related fields
    bp = db.Column(db.String(10))
    sg = db.Column(db.Float)
    al = db.Column(db.Float)
    su = db.Column(db.Float)
    rbc = db.Column(db.String(10))
    pc = db.Column(db.String(10))
    pcc = db.Column(db.String(10))
    ba = db.Column(db.String(10))
    bgr = db.Column(db.Float)
    bu = db.Column(db.Float)
    sc = db.Column(db.Float)
    sod = db.Column(db.Float)
    pot = db.Column(db.Float)
    hemo = db.Column(db.Float)
    pcv = db.Column(db.Float)
    wc = db.Column(db.String(20))
    rc = db.Column(db.Float)
    htn = db.Column(db.String(10))
    dm = db.Column(db.String(10))
    cad = db.Column(db.String(10))
    appet = db.Column(db.String(10))
    pe = db.Column(db.String(10))
    ane = db.Column(db.String(10))
    classification = db.Column(db.String(20))
    # Cardiology-related fields
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.String(10))
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.String(10))
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.String(10))
    # Diagnosis fields
    diagnosis = db.Column(db.String(50))
    radius_mean = db.Column(db.Float)
    texture_mean = db.Column(db.Float)
    # Image fields
    pneumonia_image = db.Column(db.String(200))
    brain_tumor_image = db.Column(db.String(200))
    
    
'''
class Submitdata(db.Model):
    __tablename__ = 'submitdata'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(10), unique=True, nullable=False)
    username = db.Column(db.String(100))
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.Integer)

    # Liver-related fields
    total_bilirubin = db.Column(db.Integer)
    direct_bilirubin = db.Column(db.Integer)
    alkaline_phosphotase = db.Column(db.Integer)
    alamine_aminotransferase = db.Column(db.Integer)
    aspartate_aminotransferase = db.Column(db.Integer)
    total_proteins = db.Column(db.Integer)
    albumin = db.Column(db.Integer)
    albumin_globulin_ratio = db.Column(db.Integer)

    # Kidney-related fields
    bp = db.Column(db.Integer)
    sg = db.Column(db.Integer)
    al = db.Column(db.Integer)
    su = db.Column(db.Integer)
    rbc = db.Column(db.Integer)
    pc = db.Column(db.Integer)
    pcc = db.Column(db.Integer)
    ba = db.Column(db.Integer)
    bgr = db.Column(db.Integer)
    bu = db.Column(db.Integer)
    sc = db.Column(db.Integer)
    sod = db.Column(db.Integer)
    pot = db.Column(db.Integer)
    hemo = db.Column(db.Integer)
    pcv = db.Column(db.Integer)
    wc = db.Column(db.Integer)
    rc = db.Column(db.Integer)
    htn = db.Column(db.Integer)
    dm = db.Column(db.Integer)
    cad = db.Column(db.Integer)
    appet = db.Column(db.Integer)
    pe = db.Column(db.Integer)
    ane = db.Column(db.Integer)
    classification = db.Column(db.Integer)

    # Cardiology-related fields
    cp = db.Column(db.Integer)
    trestbps = db.Column(db.Integer)
    chol = db.Column(db.Integer)
    fbs = db.Column(db.Integer)
    restecg = db.Column(db.Integer)
    thalach = db.Column(db.Integer)
    exang = db.Column(db.Integer)
    oldpeak = db.Column(db.Integer)
    slope = db.Column(db.Integer)
    ca = db.Column(db.Integer)
    thal = db.Column(db.Integer)

    # Diagnosis fields
    diagnosis = db.Column(db.Integer)
    radius_mean = db.Column(db.Integer)
    texture_mean = db.Column(db.Integer)

    # Image fields
    pneumonia_image = db.Column(db.Integer)
    brain_tumor_image = db.Column(db.Integer)


def generate_patient_id():
    return ''.join(random.choices(string.digits, k=4))

# Create all database tables
with app.app_context():
    db.create_all()
# Set up SQLAlchemy
DATABASE_URI = 'sqlite:///instance/MDDSA09.db'
engine = create_engine(DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class Dashboard:
    def __init__(self):
        self.app = app

    def generate_graph(self, disease):
        # Create a new figure
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        # Query ages and prediction status for the selected disease
        stmt = select(Prediction.age, Prediction.prediction).where(Prediction.disease == disease)
        result = session.execute(stmt)

        ages_predicted = []
        ages_non_predicted = []

        # Separate predicted and non-predicted cases
        for age, predicted in result:
            if predicted:  # If prediction exists, store in predicted list
                ages_predicted.append(age)
            else:  # If not predicted, store in non-predicted list
                ages_non_predicted.append(age)

        # Generate scatter plot with different colors
        ax.scatter(ages_predicted, [disease] * len(ages_predicted), color='red', alpha=0.5, label="Predicted")
        ax.scatter(ages_non_predicted, [disease] * len(ages_non_predicted), color='green', alpha=0.5,
                   label="Non-Predicted")

        ax.set_title(f'Age Distribution for {disease}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.legend()  # Show legend for color differentiation

        # Save to BytesIO buffer instead of file
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)  # Free memory

        # Encode the image to base64 for HTML display
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{image_base64}'

    def init_routes(self):
        @self.app.route('/dashboard', methods=['GET', 'POST'])
        def dashboard():
            image_path = None
            if request.method == 'POST':
                selected_disease = request.form.get('disease')
                if selected_disease:
                    image_path = self.generate_graph(selected_disease)

            # Fetch all distinct diseases for dropdown
            stmt = select(Prediction.disease).distinct()
            result = session.execute(stmt)
            diseases = [row[0] for row in result]

            return render_template('dashboard.html', diseases=diseases, image_path=image_path)
def main():
    dashboard = Dashboard()
    dashboard.init_routes()
    app.run(debug=True)

@app.route('/count_users', methods=['GET'])
def count_users():
    # Count the number of users in the database
    user_count = User.query.count()
    return render_template('count_users.html', user_count=user_count)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route('/')
def homeblog():
    """Render the home page."""
    return render_template('homeblog.html')

@app.route('/about1')
def about1():
    """Render the about page."""
    return render_template('about1.html')

@app.route('/contact1')
def contact1():
    """Render the contact page."""
    return render_template('contact1.html')

@app.route('/login')
def loginhome():
    """Render the login page."""
    return render_template('login.html')

@app.route('/home')
@login_required
def home():
    """Render the authenticated home page."""
    return render_template('home.html')

@app.route('/service')
def service():
    """Render the services page."""
    return render_template('services.html')

@app.route('/services')
def services():
    """Render the services page."""
    return render_template('service0.html')


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html')


# Disease information routes
@app.route('/heart-disease')
def whd():
    """Render the heart disease information page."""
    return render_template('wheart.html')


@app.route('/liver-disease')
def wld():
    """Render the liver disease information page."""
    return render_template('wliver.html')


@app.route('/kidney-disease')
def wkd():
    """Render the kidney disease information page."""
    return render_template('wkidney.html')


@app.route('/pneumonia')
def wpn():
    """Render the pneumonia information page."""
    return render_template('wpn.html')


@app.route('/breast-cancer')
def wbc():
    """Render the breast cancer information page."""
    return render_template('wbc.html')

@app.route('/braintumorw')
def wbt():
    """Render the breast cancer information page."""
    return render_template('wbt.html')
#######################################################
###Recommendations
######################################################

########
#heart

@app.route('/heartremg')
def heartremgood():
    return render_template('/rem/heart/good/heartrem.html')

@app.route('/heartfoodremg')
def heartfoodremg():
    return render_template('/rem/heart/good/heatfoodrem.html')

@app.route('/heartexeremg')
def heartexeremg():
    return render_template('/rem/heart/good/heartexerem.html')

@app.route('/heartbackg')
def heartbackg():
    return render_template('/rem/heart/good/heartrem.html')

@app.route('/heartremb')
def heartrembad():
    return render_template('/rem/heart/bad/heartrem.html')

@app.route('/heartfoodremb')
def heartfoodremb():
    return render_template('/rem/heart/bad/heatfoodrem.html')

@app.route('/heartexeremb')
def heartexeremb():
    return render_template('/rem/heart/bad/heartexerem.html')

@app.route('/heartbackb')
def heartbackb():
    return render_template('/rem/heart/bad/heartrem.html')

@app.route('/heartdocremg')
def heartdocremg():
    return render_template('/rem/heart/bad/heartremdoc.html')


#end### Heart###

@app.route('/liverremg')
def liverremg():
    return render_template('/rem/liver/good/liverrem.html')

@app.route('/liverfoodremg')
def liverfoodremg():
    return render_template('/rem/liver/good/liverfoodrem.html')

@app.route('/liverexeremg')
def liverexeremg():
    return render_template('/rem/liver/good/liverexerem.html')

@app.route('/liverbackg')
def liverbackg():
    return render_template('/rem/liver/good/liverrem.html')


@app.route('/liverremb')
def liverremb():
    return render_template('/rem/liver/bad/liverrem.html')

@app.route('/liverfoodremb')
def liverfoodremb():
    return render_template('/rem/liver/bad/liverfoodrem.html')

@app.route('/liverexeremb')
def liverexeremb():
    return render_template('/rem/liver/bad/liverexerem.html')

@app.route('/liverbackb')
def liverbackb():
    return render_template('/rem/liver/bad/liverrem.html')

@app.route('/liverremdocb')
def liverremdocb():
    return render_template('/rem/liver/bad/liverremdoc.html')
###end liver#######

@app.route('/kidneyremg')
def kidneyremg():
    return render_template('/rem/kidney/good/kidneyrem.html')

@app.route('/kidneyfoodremg')
def kidneyfoodremg():
    return render_template('/rem/kidney/good/kidneyfoodrem.html')

@app.route('/kidneyexeremg')
def kidneyexeremg():
    return render_template('/rem/kidney/good/kidneyexerem.html')

@app.route('/kidneybackg')
def kidneybackg():
    return render_template('/rem/kidney/good/kidneyrem.html')

@app.route('/kidneyremb')
def kidneyremb():
    return render_template('/rem/kidney/bad/kidneyrem.html')

@app.route('/kidneyfoodremb')
def kidneyfoodremb():
    return render_template('/rem/kidney/bad/kidneyfoodrem.html')

@app.route('/kidneyexeremb')
def kidneyexeremb():
    return render_template('/rem/kidney/bad/kidneyexerem.html')

@app.route('/kidneyremdocb')
def kidneyremdocb():
    return render_template('/rem/kidney/bad/kidneyremdoc.html')

@app.route('/kidneybackb')
def kidneybackb():
    return render_template('/rem/kidney/bad/kidneyrem.html')

####end####kidney######

@app.route('/pnerecg')
def pnerecg():
    return render_template('/rem/pneumonia/good/pnerem.html')

@app.route('/pnfoodrecg')
def pnfoodrecg():
    return render_template('/rem/pneumonia/good/pnfoodrem.html')

@app.route('/pnexerecg')
def pnexerecg():
    return render_template('/rem/pneumonia/good/pnexerem.html')

@app.route('/pnbackg')
def pnbackg():
    return render_template('/rem/pneumonia/good/pnerem.html')

@app.route('/pnerecb')
def pnerecb():
    return render_template('/rem/pneumonia/bad/pnerem.html')

@app.route('/pnfoodrecb')
def pnfoodrecb():
    return render_template('/rem/pneumonia/bad/pnfoodrem.html')

@app.route('/pnexerecb')
def pnexerecb():
    return render_template('/rem/pneumonia/bad/pnexerem.html')

@app.route('/pnerecdocb')
def pnerecdocb():
    return render_template('/rem/pneumonia/bad/pnerecdoc.html')

@app.route('/pnbackb')
def pnbackb():
    return render_template('/rem/pneumonia/bad/pnerem.html')

#####end####pn####

@app.route('/bcremdg')
def bcremdg():
    return render_template('rem/bc/good/bcrem.html')

@app.route('/bcfoodremg')
def bcfoodremg():
    return render_template('rem/bc/good/bcfoodrem.html')

@app.route('/bcexeremg')
def bcexeremg():
    return render_template('rem/bc/good/bcexerem.html')

@app.route('/bcbackg')
def bcbackg():
    return render_template('rem/bc/good/bcrem.html')


@app.route('/bcremdb')
def bcremdb():
    return render_template('rem/bc/bad/bcrem.html')

@app.route('/bcfoodremb')
def bcfoodremb():
    return render_template('rem/bc/bad/bcfoodrem.html')

@app.route('/bcexeremb')
def bcexeremb():
    return render_template('rem/bc/bad/bcexerem.html')

@app.route('/bcremddocb')
def bcremddocb():
    return render_template('rem/bc/bad/bcremddoc.html')

@app.route('/bcbackb')
def bcbackb():
    return render_template('rem/bc/bad/bcrem.html')

###end##bc#######

@app.route('/btremdg')
def btremdg():
    return render_template('rem/braintumor/good/braintumorrem.html')

@app.route('/btfoodremg')
def btfoodremg():
    return render_template('rem/braintumor/good/braintumorfoodrem.html')

@app.route('/btexeremg')
def btexeremg():
    return render_template('/rem/braintumor/good/braintumorexerem.html')

@app.route('/btbackg')
def btbackg():
    return render_template('rem/braintumor/good/braintumorrem.html')


@app.route('/btremdb')
def btremdb():
    return render_template('rem/braintumor/bad/braintumorrem.html')

@app.route('/btfoodremb')
def btfoodremb():
    return render_template('rem/braintumor/bad/braintumorfoodrem.html')

@app.route('/btexeremb')
def btexeremb():
    return render_template('rem/braintumor/bad/braintumorexerem.html')

@app.route('/btdocremb')
def btdocremb():
    return render_template('rem/braintumor/bad/braintumordoctorrem.html')

@app.route('/btbackb')
def btbackb():
    return render_template('rem/braintumor/bad/braintumorrem.html')

###end##bt############
#####################################################################################################

@app.route('/update')
def update():
    return render_template('update.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for('homeblog'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone_number = request.form.get('phonenumber')
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email format')
            return redirect(url_for('register'))

        # Validate phone number (e.g., check if it's numeric and has 10-15 digits)
        if not re.match(r"^\d{10}$", phone_number):
            flash('Invalid phone number. It must be 10 digits long.')
            return redirect(url_for('register'))

        # Validate password constraints
        if not validate_password(password):
            flash(
                'Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        existing_phone = User.query.filter_by(phone_number=phone_number).first()

        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))
        if existing_email:
            flash('Email address already exists')
            return redirect(url_for('register'))
        if existing_phone:
            flash('Phone number already exists')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, phone_number=phone_number, name=name)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


def validate_password(password):
    """Check password complexity requirements."""
    if (len(password) < 8 or
            not re.search(r"[A-Z]", password) or  # At least one uppercase letter
            not re.search(r"[a-z]", password) or  # At least one lowercase letter
            not re.search(r"[0-9]", password) or  # At least one digit
            not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):  # At least one special character
        return False
    return True


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle password reset request."""
    if request.method == 'POST':
        username = request.form.get('username')
        user = User.query.filter_by(username=username).first()

        if user:
            # If the user exists, render the password reset form
            return render_template('reset_password.html', user=user)
        else:
            flash('Username not found.', 'error')

    return render_template('forgot_password.html')




@app.route('/update-password', methods=['POST'])
def update_password():
    """Update the user's password."""
    user_id = request.form.get('user_id')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # Fetch the user from the database
    user = User.query.get(user_id)

    # Check if the user exists
    if not user:
        flash('User not found. Unable to update password.', 'error')
        return render_template('reset_password.html')  # Redirect to reset password page

    # Check if the new password is valid
    if not validate_password(new_password):
        flash('Password must be at least 8 characters long and include at least one uppercase letter, one lowercase letter, one digit, and one special character.', 'error')
        return render_template('reset_password.html', user=user)

    # Check if the new password is the same as the current password
    if user.check_password(new_password):
        flash('New password cannot be the same as the current password.', 'error')
        return render_template('reset_password.html', user=user)

    # Check if the new password matches the confirm password
    if new_password != confirm_password:
        flash('Passwords do not match.', 'error')
        return render_template('reset_password.html', user=user)

    # If all checks pass, update the password
    user.set_password(new_password)  # Hash the new password
    db.session.commit()
    flash('Your password has been updated successfully. You can now log in.', 'success')
    return redirect(url_for('login'))


def check_password(self, password):
    """Check if the provided password matches the stored password."""
    return check_password_hash(self.password_hash, password)

def set_password(self, password):
        """Set the user's password (hash it)."""
        self.password_hash = generate_password_hash(password)


# Load the trained models
MODEL_PATHS = {
    'heart': 'models/updated/heart.pkl',
    'breast_cancer': 'models/updated/breast_cancer.pkl',
    'kidney': 'models/updated/kidney.pkl',
    'liver': 'models/updated/liver.pkl',
    'pneumonia': 'models/updated/pn.h5',
    'braintumor':'models/updated/braintumor.h5'
}

models = {}


def load_models():
    for name, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            if name == 'pneumonia' or name == 'braintumor':
                models[name] = tf.keras.models.load_model(path)
            else:
                models[name] = joblib.load(path)
            logging.info(f"{name.capitalize()} model loaded successfully from {path}")
        except Exception as e:
            logging.error(f"Error loading {name} model: {str(e)}", exc_info=True)
            models[name] = None


load_models()


def create_pdf(data, prediction, disease):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add watermark
    def add_watermark(canvas):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 100)  # Bigger font size for visibility
        canvas.setFillColor(Color(0.7, 0.7, 0.7, alpha=0.3))  # Light gray with transparency
        canvas.translate(width / 2, height / 2)
        canvas.rotate(45)  # Rotate text diagonally
        canvas.drawCentredString(0, 0, "MDDS")
        canvas.restoreState()
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "MDDS: Multi Disease Detection System")
    c.drawString(100, height - 80, f"{disease.capitalize()} Prediction Report")
    # Input data
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 160, "Input Data:")
    c.setFont("Helvetica", 12)
    y = height - 180
    for key, value in data.items():
        c.drawString(120, y, f"{key}: {value}")
        y -= 20
    # Draw the prediction result right after the data
    c.setFont("Helvetica-Bold", 12)
    y -= 20  # Adjust spacing before the prediction result
    c.drawString(120, y, f"Prediction: {prediction}")
    # Warning message
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 100, "Important Notice:")
    c.setFont("Helvetica", 10)
    c.drawString(100, 80, "This prediction is based on a machine learning model and should not be")
    c.drawString(100, 65, "considered as a definitive medical diagnosis. Please consult with a healthcare")
    c.drawString(100, 50, "professional for accurate medical advice and proper diagnosis.")
    add_watermark(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    try:
        age = int(data.get("age", 0))
        gender = data.get("gender", "Unknown")
        new_entry = Prediction(age=age, gender=gender, disease=disease, prediction=prediction)
        db.session.add(new_entry)
        db.session.commit()
        logging.info(f"Data saved to Predictions database: {new_entry}")
    except Exception as e:
        logging.error(f"Error saving to Predictions database: {e}")
        db.session.rollback()
    return buffer


# Save to Predictions Database
def save_to_predictions_db(data, prediction, disease):
    pass



@app.route('/predict/<disease>', methods=['GET'])
def predict_form(disease):
    if disease not in MODEL_PATHS:
        return "Invalid disease type", 404
    return render_template(f'{disease}.html')


@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        if disease not in models:
            raise ValueError(f"Invalid disease type: {disease}")

        model = models[disease]
        if model is None:
            raise ValueError(f"{disease.capitalize()} model not loaded. Cannot make predictions.")

        # Handle form data for non-pneumonia diseases
        if disease != 'pneumonia':
            form_data = {key: request.form.get(key) for key in request.form if key != 'disease'}
            logging.info(f"Received data for {disease}: {form_data}")

            # Convert form data to floats
            try:
                form_data = {key: float(value) for key, value in form_data.items()}
            except ValueError as ve:
                logging.error(f"Error converting form data to float: {str(ve)}")
                return jsonify({"error": "Invalid data format"}), 400

            data = np.array(list(form_data.values())).reshape(1, -1)

        # Handle file upload for pneumonia
        else:
            if 'image' not in request.files:
                logging.error("No image file provided for pneumonia prediction.")
                return jsonify({"error": "No image file provided"}), 400

            image_file = request.files['image']
            if image_file.filename == '':
                logging.error("Empty image file name for pneumonia prediction.")
                return jsonify({"error": "No image selected"}), 400

            # Process the image
            try:
                img = Image.open(image_file).convert('RGB')  # Ensure image is in RGB mode
                img = img.resize((300, 300))  # Resize to match the input shape
                img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
                data = img_array.reshape(1, 300, 300, 3)  # Reshape for model input
            except Exception as e:
                logging.error(f"Error processing image file: {str(e)}")
                return jsonify({"error": "Invalid image file"}), 400

            form_data = {"image": image_file.filename}  # Log the filename

        logging.debug(f"Data shape for {disease} model: {data.shape}")

        # Make prediction
        prediction = model.predict(data)

        # Define result based on disease type
        if disease == 'heart':
            result = 'Heart disease detected' if prediction[0] == 1 else 'No heart disease detected'
        elif disease == 'breast_cancer':
            result = 'You have cancer' if prediction[0] == 1 else 'You don\'t have cancer'
        elif disease == 'kidney':
            result = 'Chronic Kidney Disease detected' if prediction[0] == 1 else 'No Chronic Kidney Disease detected'
        elif disease == 'liver':
            result = 'Liver disease detected' if prediction[0] == 1 else 'No liver disease detected'
        elif disease == 'pneumonia':
            result = 'Pneumonia detected' if prediction[0][0] > 0.5 else 'No pneumonia detected'

        logging.info(f"{disease.capitalize()} prediction made: {result}")

        # Generate and send PDF
        pdf_buffer = create_pdf(form_data, result, disease)
        logging.debug(f"PDF generated successfully for {disease}. Buffer size: {pdf_buffer.getbuffer().nbytes} bytes")

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'{disease}_report.pdf',
            mimetype='application/pdf'
        )

    except ValueError as ve:
        logging.error(f"Invalid input for {disease}: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Prediction error for {disease}: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route("/predict/braintumor", methods=["POST"])
def predict_brain_tumor():
    try:
        # Ensure the model is loaded
        if 'braintumor' not in models or models['braintumor'] is None:
            raise ValueError("Brain Tumor model not loaded. Cannot make predictions.")

        model = models['braintumor']

        # Check for an uploaded image file
        if 'image' not in request.files:
            logging.error("No image file provided for brain tumor prediction.")
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            logging.error("Empty image file name for brain tumor prediction.")
            return jsonify({"error": "No image selected"}), 400

        # Process the uploaded image
        try:
            img = Image.open(image_file).convert('RGB')  # Convert to RGB
            img = img.resize((300, 300))  # Resize to match model input
            img_array = np.array(img) / 255.0  # Normalize pixel values
            data = img_array.reshape(1, 300, 300, 3)  # Reshape for model input
        except Exception as e:
            logging.error(f"Error processing image file: {str(e)}")
            return jsonify({"error": "Invalid image file"}), 400

        logging.debug(f"Data shape for brain tumor model: {data.shape}")

        # Make prediction
        prediction = model.predict(data)
        result = 'No Brain Tumor detected' if prediction[0][0] > 0.5 else 'Brain tumor detected'

        logging.info(f"Brain tumor prediction made: {result}")

        # Generate and send PDF report
        form_data = {"image": image_file.filename}
        pdf_buffer = create_pdf(form_data, result, 'braintumor')
        logging.debug(f"PDF generated successfully for brain tumor. Buffer size: {pdf_buffer.getbuffer().nbytes} bytes")

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='brain_tumor_report.pdf',
            mimetype='application/pdf'
        )

    except ValueError as ve:
        logging.error(f"Invalid input for brain tumor: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Prediction error for brain tumor: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

# Error handling
@app.errorhandler(404)
def page_not_found(error):
    """Render 404 error page."""
    return render_template('408.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Render 500 error page."""
    return render_template('500.html'), 500
#updates from here


@app.route('/reg')
def reg():
    existing_entry = Submitdata.query.filter_by(username=current_user.username).first()
    if existing_entry:
        # If you want to show the form but let them know they have an entry:
        return render_template('html.html', message='You already have an existing entry. Only one entry per user is allowed.')
    # Generate a unique patient ID
    patient_id = generatepatient_id()
    return render_template('html.html', patient_id=patient_id,username=current_user.username)


# Helper function to generate random patient ID
def generatepatient_id():
    return ''.join(random.choices(string.digits, k=4))


@app.route('/get_patient_data', methods=['POST'])
def get_patient_data():
    """
    This route receives a JSON body containing "patient_id".
    It returns the patient's data as JSON if found, or an error if not.
    """
    data = request.json
    patient_id = data.get('patient_id')

    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400

    patient = Submitdata.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    # Return the fields that match the HTML form's 'name' attributes
    return jsonify({
        "patient_id": patient.patient_id,
        "name": patient.name,
        "username": patient.username,
        "age": patient.age,
        "gender": patient.gender,
        "total_bilirubin": patient.total_bilirubin,
        "direct_bilirubin": patient.direct_bilirubin,
        "alkaline_phosphotase": patient.alkaline_phosphotase,
        "alamine_aminotransferase": patient.alamine_aminotransferase,
        "aspartate_aminotransferase": patient.aspartate_aminotransferase,
        "total_proteins": patient.total_proteins,
        "albumin": patient.albumin,
        "albumin_globulin_ratio": patient.albumin_globulin_ratio,
        "bp": patient.bp,
        "sg": patient.sg,
        "al": patient.al,
        "su": patient.su,
        "rbc": patient.rbc,
        "pc": patient.pc,
        "pcc": patient.pcc,
        "ba": patient.ba,
        "bgr": patient.bgr,
        "bu": patient.bu,
        "sc": patient.sc,
        "sod": patient.sod,
        "pot": patient.pot,
        "hemo": patient.hemo,
        "pcv": patient.pcv,
        "wc": patient.wc,
        "rc": patient.rc,
        "htn": patient.htn,
        "dm": patient.dm,
        "cad": patient.cad,
        "appet": patient.appet,
        "pe": patient.pe,
        "ane": patient.ane,
        "classification": patient.classification,
        "cp": patient.cp,
        "trestbps": patient.trestbps,
        "chol": patient.chol,
        "fbs": patient.fbs,
        "restecg": patient.restecg,
        "thalach": patient.thalach,
        "exang": patient.exang,
        "oldpeak": patient.oldpeak,
        "slope": patient.slope,
        "ca": patient.ca,
        "thal": patient.thal,
        "diagnosis": patient.diagnosis,
        "radius_mean": patient.radius_mean,
        "texture_mean": patient.texture_mean,
        "pneumonia_image": patient.pneumonia_image,
        "brain_tumor_image": patient.brain_tumor_image
    })
'''
@app.route('/get_patient_data', methods=['POST'])
def get_patient_data():
    data = request.json
    patient_id = data.get('patient_id')

    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400

    patient = Submitdata.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    # Print patient data for debugging
    response_data = {
        "patient_id": patient.patient_id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "cp": patient.cp,
        "trestbps": patient.trestbps,
        "chol": patient.chol,
        "fbs": patient.fbs,
        "restecg": patient.restecg,
        "thalach": patient.thalach,
        "exang": patient.exang,
        "oldpeak": patient.oldpeak,
        "slope": patient.slope,
        "ca": patient.ca,
        "thal": patient.thal
    }

    print("Fetched Data:", response_data)  # Debugging
    return jsonify(response_data)
'''

'''

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        data = request.form.to_dict()

        # Get the username of the currently logged-in user
        if not current_user.is_authenticated:
            return render_template('html.html', message='User must be logged in to submit data')

        username = current_user.username  # Get username from current logged-in user

        # Generate unique patient ID
        patient_id = generate_patient_id()
        while Submitdata.query.filter_by(patient_id=patient_id).first():
            patient_id = generate_patient_id()

        # Handle file uploads
        pneumonia_image = request.files.get('pneumonia_image')
        brain_tumor_image = request.files.get('brain_tumor_image')

        pneumonia_image_path = None
        brain_tumor_image_path = None

        if pneumonia_image:
            filename = secure_filename(pneumonia_image.filename)
            pneumonia_image_path = os.path.join(app.config['UPLOAD'], filename)
            pneumonia_image.save(pneumonia_image_path)

        if brain_tumor_image:
            filename = secure_filename(brain_tumor_image.filename)
            brain_tumor_image_path = os.path.join(app.config['UPLOAD'], filename)
            brain_tumor_image.save(brain_tumor_image_path)

        # Create new record in the Submitdata table
        new_data = Submitdata(
            patient_id=patient_id,
            name=data.get('name'),  # Patient's full name
            username=username,  # Using logged-in user's username
            age=int(data.get('age', 0)),
            gender = 1 if data.get('gender') == 'male' else 0,
            total_bilirubin=float(data.get('total_bilirubin', 0)),
            direct_bilirubin=float(data.get('direct_bilirubin', 0)),
            alkaline_phosphotase=float(data.get('alkaline_phosphotase', 0)),
            alamine_aminotransferase=float(data.get('alamine_aminotransferase', 0)),
            aspartate_aminotransferase=float(data.get('aspartate_aminotransferase', 0)),
            total_proteins=float(data.get('total_proteins', 0)),
            albumin=float(data.get('albumin', 0)),
            albumin_globulin_ratio=float(data.get('albumin_globulin_ratio', 0)),
            bp=data.get('bp'),
            sg=float(data.get('sg', 0)),
            al=float(data.get('al', 0)),
            su=float(data.get('su', 0)),
            rbc=data.get('rbc'),
            pc=data.get('pc'),
            pcc=data.get('pcc'),
            ba=data.get('ba'),
            bgr=float(data.get('bgr', 0)),
            bu=float(data.get('bu', 0)),
            sc=float(data.get('sc', 0)),
            sod=float(data.get('sod', 0)),
            pot=float(data.get('pot', 0)),
            hemo=float(data.get('hemo', 0)),
            pcv=float(data.get('pcv', 0)),
            wc=data.get('wc'),
            rc=float(data.get('rc', 0)),
            htn=data.get('htn'),
            dm=data.get('dm'),
            cad=data.get('cad'),
            appet=data.get('appet'),
            pe=data.get('pe'),
            ane=data.get('ane'),
            classification=data.get('classification'),
            cp=float(data.get('cp', 0)),
            trestbps=float(data.get('trestbps', 0)),
            chol=float(data.get('chol', 0)),
            fbs=data.get('fbs'),
            restecg=float(data.get('restecg', 0)),
            thalach=float(data.get('thalach', 0)),
            exang=data.get('exang'),
            oldpeak=float(data.get('oldpeak', 0)),
            slope=float(data.get('slope', 0)),
            ca=float(data.get('ca', 0)),
            thal=data.get('thal'),
            target=data.get('target'),
            diagnosis=data.get('diagnosis'),
            radius_mean=float(data.get('radius_mean', 0)),
            texture_mean=float(data.get('texture_mean', 0)),
            pneumonia_image=pneumonia_image_path,
            brain_tumor_image=brain_tumor_image_path
        )

        db.session.add(new_data)
        db.session.commit()

        return render_template('html.html', message="Data submitted successfully!", patient_id=patient_id)

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400
'''

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        data = request.form.to_dict()

        # Ensure user is logged in
        if not current_user.is_authenticated:
            return render_template('html.html', message='User must be logged in to submit data')

        username = current_user.username  # Get username from current logged-in user

        # Generate unique patient ID
        patient_id = generate_patient_id()
        while Submitdata.query.filter_by(patient_id=patient_id).first():
            patient_id = generate_patient_id()

        # Handle file uploads
        pneumonia_image = request.files.get('pneumonia_image')
        brain_tumor_image = request.files.get('brain_tumor_image')

        pneumonia_image_path = None
        brain_tumor_image_path = None

        if pneumonia_image:
            filename = secure_filename(pneumonia_image.filename)
            pneumonia_image_path = os.path.join(app.config['UPLOAD'], filename)
            pneumonia_image.save(pneumonia_image_path)

        if brain_tumor_image:
            filename = secure_filename(brain_tumor_image.filename)
            brain_tumor_image_path = os.path.join(app.config['UPLOAD'], filename)
            brain_tumor_image.save(brain_tumor_image_path)

        # Create new record in the Submitdata table
        new_data = Submitdata(
            patient_id=patient_id,
            name=data.get('name', ""),  # Default to empty string if missing
            username=username,  # Logged-in user's username
            age=int(data.get('age', 0)),
            gender=1 if data.get('gender', "").lower() == 'male' else 0,  # 1 for male, 0 for female
            total_bilirubin=int(data.get('total_bilirubin', 0)),
            direct_bilirubin=int(data.get('direct_bilirubin', 0)),
            alkaline_phosphotase=int(data.get('alkaline_phosphotase', 0)),
            alamine_aminotransferase=int(data.get('alamine_aminotransferase', 0)),
            aspartate_aminotransferase=int(data.get('aspartate_aminotransferase', 0)),
            total_proteins=int(data.get('total_proteins', 0)),
            albumin=int(data.get('albumin', 0)),
            albumin_globulin_ratio=int(data.get('albumin_globulin_ratio', 0)),
            bp=int(data.get('bp', 0)),
            sg=int(data.get('sg', 0)),
            al=int(data.get('al', 0)),
            su=int(data.get('su', 0)),
            rbc=int(data.get('rbc', 0)),
            pc=int(data.get('pc', 0)),
            pcc=int(data.get('pcc', 0)),
            ba=int(data.get('ba', 0)),
            bgr=int(data.get('bgr', 0)),
            bu=int(data.get('bu', 0)),
            sc=int(data.get('sc', 0)),
            sod=int(data.get('sod', 0)),
            pot=int(data.get('pot', 0)),
            hemo=int(data.get('hemo', 0)),
            pcv=int(data.get('pcv', 0)),
            wc=int(data.get('wc', 0)),
            rc=int(data.get('rc', 0)),
            htn=int(data.get('htn', 0)),
            dm=int(data.get('dm', 0)),
            cad=int(data.get('cad', 0)),
            appet=int(data.get('appet', 0)),
            pe=int(data.get('pe', 0)),
            ane=int(data.get('ane', 0)),
            classification=int(data.get('classification', 0)),
            cp=int(data.get('cp', 0)),
            trestbps=int(data.get('trestbps', 0)),
            chol=int(data.get('chol', 0)),
            fbs=int(data.get('fbs', 0)),
            restecg=int(data.get('restecg', 0)),
            thalach=int(data.get('thalach', 0)),
            exang=int(data.get('exang', 0)),
            oldpeak=int(data.get('oldpeak', 0)),
            slope=int(data.get('slope', 0)),
            ca=int(data.get('ca', 0)),
            thal=int(data.get('thal', 0)),
            diagnosis=int(data.get('diagnosis', 0)),
            radius_mean=int(data.get('radius_mean', 0)),
            texture_mean=int(data.get('texture_mean', 0)),
            pneumonia_image=pneumonia_image_path,
            brain_tumor_image=brain_tumor_image_path
        )

        db.session.add(new_data)
        db.session.commit()
        return render_template('html.html', message='Data successfully submitted')

    except Exception as e:
        print("Error:", str(e))
        return render_template('html.html', message='An error occurred while processing the request')



@app.route('/update/<patient_id>', methods=['GET', 'POST'])
def update_form(patient_id):
    """
    This route handles both:
      - GET request: Displaying the update page (optionally showing current data)
      - POST request: Updating the patient's data in the database
    """
    try:
        # 1. Find existing record by patient_id
        existing_data = Submitdata.query.filter_by(patient_id=patient_id).first()
        if not existing_data:
            # You can choose to render update.html with an error message or just return a message
            return render_template('update.html', message="No record found with this Patient ID")

        # 2. If it's a POST, we update the record
        if request.method == 'POST':
            form_data = request.form.to_dict()

            # Update fields from the form
            existing_data.name = form_data.get('name')
            existing_data.age = int(form_data.get('age', 0))
            existing_data.gender = 1 if form_data.get('gender') == 'male' else 0
            existing_data.gender = form_data.get('gender')

            # -----------------------------
            # LIVER FIELDS
            # -----------------------------
            existing_data.total_bilirubin = int(form_data.get('total_bilirubin', 0))
            existing_data.direct_bilirubin = int(form_data.get('direct_bilirubin', 0))
            existing_data.alkaline_phosphotase = int(form_data.get('alkaline_phosphotase', 0))
            existing_data.albumin = int(form_data.get('albumin', 0))
            existing_data.alamine_aminotransferase = int(form_data.get('alamine_aminotransferase', 0))
            existing_data.aspartate_aminotransferase = int(form_data.get('aspartate_aminotransferase', 0))
            existing_data.total_proteins = int(form_data.get('total_proteins', 0))
            existing_data.albumin_and_globulin_ratio = int(form_data.get('albumin_and_globulin_ratio', 0))

            # -----------------------------
            # KIDNEY (CKD) FIELDS
            # -----------------------------
            existing_data.bp = form_data.get('bp')
            existing_data.sg = int(form_data.get('sg', 0))
            existing_data.al = int(form_data.get('al', 0))
            existing_data.bgr = int(form_data.get('bgr', 0))
            existing_data.rbc = form_data.get('rbc')
            existing_data.rbc_count = int(form_data.get('rbc_count', 0))
            existing_data.pc = form_data.get('pc')
            existing_data.pcc = form_data.get('pcc')
            existing_data.ba = form_data.get('ba')
            existing_data.bu = int(form_data.get('bu', 0))  # blood urea
            existing_data.sc = int(form_data.get('sc', 0))  # serum creatinine
            existing_data.sod = int(form_data.get('sod', 0))
            existing_data.pot = int(form_data.get('pot', 0))
            existing_data.hemo = int(form_data.get('hemo', 0))
            existing_data.pcv = int(form_data.get('pcv', 0))
            existing_data.wc = int(form_data.get('wc', 0))
            existing_data.rc = int(form_data.get('rc', 0))

            # -----------------------------
            # HEART FIELDS
            # -----------------------------
            existing_data.cp = int(form_data.get('cp', 0))
            existing_data.trestbps = int(form_data.get('trestbps', 0))
            existing_data.chol = int(form_data.get('chol', 0))
            existing_data.fbs = form_data.get('fbs')
            existing_data.thalach = int(form_data.get('thalach', 0))
            existing_data.exang = form_data.get('exang')

            # If your form includes these additional heart-related fields:
            existing_data.restecg = int(form_data.get('restecg', 0))
            existing_data.oldpeak = int(form_data.get('oldpeak', 0))
            existing_data.slope = int(form_data.get('slope', 0))
            existing_data.ca = int(form_data.get('ca', 0))
            existing_data.thal = int(form_data.get('thal', 0))

            # Handle file uploads
            pneumonia_image = request.files.get('pneumonia_image')
            brain_tumor_image = request.files.get('brain_tumor_image')

            if pneumonia_image and pneumonia_image.filename:
                filename = secure_filename(pneumonia_image.filename)
                pneumonia_image_path = os.path.join(app.config['UPLOAD'], filename)
                pneumonia_image.save(pneumonia_image_path)
                existing_data.pneumonia_image = pneumonia_image_path

            if brain_tumor_image and brain_tumor_image.filename:
                filename = secure_filename(brain_tumor_image.filename)
                brain_tumor_image_path = os.path.join(app.config['UPLOAD'], filename)
                brain_tumor_image.save(brain_tumor_image_path)
                existing_data.brain_tumor_image = brain_tumor_image_path

            # Save changes
            db.session.commit()

            # Render the same page or redirect, showing success message
            return render_template('update.html', message="Data updated successfully!", patient_id=patient_id)

        # 3. If GET, optionally show the update page
        # In many apps, you'd just do 'return render_template("update.html", data=existing_data)'
        # to pre-populate a form. But here, we rely on the JavaScript search approach.
        return render_template('update.html')

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400


##Diet Recommendation


class RecipeProcessor:
    @staticmethod
    def parse_time(time_str):
        """Parse ISO duration format into minutes"""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return 0, 0

        time_str = time_str.replace("PT", "")
        total_minutes = 0

        if 'H' in time_str:
            hours = int(time_str.split('H')[0])
            total_minutes += hours * 60
            time_str = time_str.split('H')[1]

        if 'M' in time_str:
            minutes = int(time_str.split('M')[0])
            total_minutes += minutes

        return divmod(total_minutes, 60)

    @staticmethod
    def parse_list(text):
        """Parse ingredient lists and quantities"""
        if pd.isna(text):
            return []
        cleaned = text.replace('c(', '').replace(')', '').strip()
        return [item.strip(' "\'') for item in cleaned.split(',') if item.strip()]

    @staticmethod
    def format_ingredients(ingredients, quantities):
        """Combine ingredients with their quantities"""
        if not ingredients or not quantities:
            return []
        return [f"{qty} {ing}" for qty, ing in zip(quantities, ingredients)]

    @staticmethod
    def calculate_nutrition_score(calories, target_calories):
        """Calculate how well recipe matches target calories (0-100 score)"""
        difference = abs(calories - target_calories)
        max_difference = target_calories
        score = max(0, 100 - (difference / max_difference * 100))
        return round(score, 1)


class RecipeRecommender:
    def __init__(self):
        self.processor = RecipeProcessor()

    def load_dataset(self):
        """Load and preprocess recipe dataset"""
        try:
            data = pd.read_csv('recipes.csv')

            for time_field in ['CookTime', 'PrepTime', 'TotalTime']:
                data[f'{time_field}Hours'], data[f'{time_field}Minutes'] = zip(
                    *data[time_field].apply(self.processor.parse_time)
                )

            data['RecipeIngredientParts'] = data['RecipeIngredientParts'].apply(self.processor.parse_list)
            data['RecipeIngredientQuantities'] = data['RecipeIngredientQuantities'].apply(self.processor.parse_list)
            data['Calories'] = pd.to_numeric(data['Calories'], errors='coerce').round(1)
            data = data.dropna(subset=['Calories'])
            data['Keywords'] = data['Keywords'].apply(self.processor.parse_list)

            return data
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None

    def get_recommendations(self, target_calories, num_recipes=3):
        """Get recipe recommendations based on calorie target"""
        data = self.load_dataset()
        if data is None:
            return None

        try:
            data['CalorieDifference'] = abs(data['Calories'] - target_calories)
            data['NutritionScore'] = data.apply(
                lambda x: self.processor.calculate_nutrition_score(x['Calories'], target_calories),
                axis=1
            )

            closest_matches = data.nsmallest(num_recipes, 'CalorieDifference')

            recommendations = []
            for _, recipe in closest_matches.iterrows():
                processed_recipe = {
                    'RecipeCategory': recipe['RecipeCategory'],
                    'Calories': recipe['Calories'],
                    'CalorieDifference': recipe['CalorieDifference'],
                    'NutritionScore': recipe['NutritionScore'],
                    'PrepTimeHours': int(recipe['PrepTimeHours']),
                    'PrepTimeMinutes': int(recipe['PrepTimeMinutes']),
                    'CookTimeHours': int(recipe['CookTimeHours']),
                    'CookTimeMinutes': int(recipe['CookTimeMinutes']),
                    'TotalTimeHours': int(recipe['TotalTimeHours']),
                    'TotalTimeMinutes': int(recipe['TotalTimeMinutes']),
                    'Ingredients': self.processor.format_ingredients(
                        recipe['RecipeIngredientParts'],
                        recipe['RecipeIngredientQuantities']
                    ),
                    'Instructions': self.processor.parse_list(recipe['RecipeInstructions']),
                    'Keywords': recipe['Keywords']
                }
                recommendations.append(processed_recipe)

            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None


class HealthForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    height = IntegerField('Height (cm)', validators=[DataRequired()])
    weight = IntegerField('Weight (kg)', validators=[DataRequired()])
    activity_level = SelectField('Activity Level', choices=[
        ('sedentary', 'Sedentary'),
        ('light', 'Light'),
        ('moderate', 'Moderate'),
        ('active', 'Active'),
        ('very_active', 'Very Active')
    ], validators=[DataRequired()])
    goal = SelectField('Weight Goal', choices=[
        ('maintain', 'Maintain Weight'),
        ('mild_loss', 'Mild Weight Loss'),
        ('loss', 'Weight Loss'),
        ('extreme_loss', 'Extreme Weight Loss'),
        ('gain', 'Weight Gain')
    ], validators=[DataRequired()])
    num_meals = IntegerField('Number of Meals per Day (1-5)', validators=[
        DataRequired(),
        NumberRange(min=1, max=5, message="Please choose a number between 1 and 5")
    ])
    submit = SubmitField('Calculate and Find Recipes')


# Initialize recommender
recommender = RecipeRecommender()

@app.route('/diet', methods=['GET', 'POST'])
def diet():
    form = HealthForm()
    if form.validate_on_submit():
        # Calculate BMR using Mifflin-St Jeor Equation
        age = form.age.data
        gender = form.gender.data
        height = form.height.data
        weight = form.weight.data
        activity_level = form.activity_level.data
        goal = form.goal.data
        num_meals = form.num_meals.data

        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Activity level adjustments
        activity_factors = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        tdee = bmr * activity_factors[activity_level]

        # Goal adjustments
        goal_factors = {
            'maintain': 1.0,
            'mild_loss': 0.9,
            'loss': 0.8,
            'extreme_loss': 0.7,
            'gain': 1.1
        }
        adjusted_tdee = tdee * goal_factors[goal]

        # Calculate BMI
        bmi = weight / (height / 100) ** 2

        # BMI Classification
        if bmi < 18.5:
            bmi_classification = "Underweight"
        elif 18.5 <= bmi < 24.9:
            bmi_classification = "Normal"
        elif 25 <= bmi < 29.9:
            bmi_classification = "Overweight"
        else:
            bmi_classification = "Obesity"

        # Calculate calories per meal
        calories_per_meal = adjusted_tdee / num_meals

        # Get recipe recommendations
        recipes = recommender.get_recommendations(calories_per_meal)

        # Round values
        tdee = round(adjusted_tdee, 1)
        bmi = round(bmi, 1)
        calories_per_meal = round(calories_per_meal, 1)

        return render_template('cal.html',
                               form=form,
                               tdee=tdee,
                               bmi=bmi,
                               bmi_classification=bmi_classification,
                               calories_per_meal=calories_per_meal,
                               num_meals=num_meals,
                               recipes=recipes)

    return render_template('cal.html', form=form)


@app.template_filter('format_time')
def format_time(hours, minutes):
    """Template filter to format time nicely"""
    if hours == 0:
        return f"{minutes}m"
    return f"{hours}h {minutes}m"

@app.route('/get_patient_id/<username>', methods=['GET'])
def get_patient_id(username):
    user_entry = Submitdata.query.filter_by(username=username).first()
    if user_entry:
        return jsonify({"patient_id": user_entry.patient_id})
    return jsonify({"error": "Patient ID not found"}), 404

@app.route('/find_patient', methods=['GET'])
def find_patient():
    return render_template('getpatientid.html')



if __name__ == '__main__':
    main()
    app.run(debug=True)
