import os,argparse
from flask import Flask, jsonify, request
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import pytesseract

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'images'
@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/endpoint', methods=['POST'])

def endpoint():
    # Vérifier si les fichiers sont présents
    if 'image_1' not in request.files or 'image_2' not in request.files:
        return 'No file uploaded', 400

    # Récupérer les fichiers
    image_1 = request.files['image_1']
    image_2 = request.files['image_2']
    
    # Enregistrer les fichiers sur le serveur
    filename_1 = secure_filename(image_1.filename)
    filename_2 = secure_filename(image_2.filename)
    image_1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_1))
    image_2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_2))
    # Ouvrir l'image JPEG
    document1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename_1))
    document2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename_2))
    signatures1 = extract_signatures(document1)
    signatures2 = extract_signatures(document2)
    signature_pairs = zip(signatures1, signatures2)
    threshold = 0.1
    bool=True
    for i, (signature1, signature2) in enumerate(signature_pairs):
    # Calculer les moments de Hu pour chaque signature
        gray1 = cv2.cvtColor(signature1, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moments1 = cv2.HuMoments(cv2.moments(contours1[0])).flatten()
        gray2 = cv2.cvtColor(signature2, cv2.COLOR_BGR2GRAY)
        ret, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moments2 = cv2.HuMoments(cv2.moments(contours2[0])).flatten()
    # Comparer les moments de Hu des deux signatures
        distance = np.linalg.norm(moments1 - moments2)
        if distance < threshold:
            print("La paire de signatures", i+1, "correspond")
           
        else:
            print("La paire de signatures", i+1, "ne correspond pas")
            bool=False
           
    if(bool==False) :
      return jsonify({'statut':400})
    else:
      return jsonify({'statut': 200})


def extract_signatures(document):
    signatures = []
    # Convertir l'image en niveaux de gris pour une meilleure détection des contours
    gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    # Appliquer un seuillage pour convertir l'image en noir et blanc
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Trouver tous les contours dans l'image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Parcourir chaque contour et vérifier s'il ressemble à une signature
    for contour in contours:
        # Récupérer les coordonnées du rectangle englobant le contour
        x, y, w, h = cv2.boundingRect(contour)
        # Vérifier si la hauteur et la largeur du rectangle sont raisonnables pour une signature
        if h > 50 and w > 150 and h < 300 and w < 500:
            # Extraire la signature de l'image originale en utilisant les coordonnées du rectangle
            signature = document[y:y+h, x:x+w]
            signatures.append(signature)
    return signatures
@app.route("/test")
def test():
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_path = r"images\testS.jpeg"
    img = Image.open(image_path)
    gray = img.convert('L')
    blurred = img.filter(ImageFilter.BLUR)
    thresholded = img.convert('1')
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(thresholded)
    image_path = r"images\testS.jpeg"
    image_retouchee = numeriser_image(image_path)
    image_retouchee.save('image_retouchee.png')
# Afficher l'image retouchée
    image_retouchee.show()
    return jsonify(text)



def numeriser_image(image_path):
    # Charger l'image
    img = Image.open(image_path)

    # Conversion en niveaux de gris
    gray = img.convert('L')

    # Amélioration du contraste de l'image
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_img = enhancer.enhance(2.0)  # Ajustez la valeur 2.0 selon vos besoins

    # Binarisation de l'image
    threshold = 128
    binarized_img = enhanced_img.point(lambda p: 255 if p > threshold else 0, '1')

    return binarized_img






