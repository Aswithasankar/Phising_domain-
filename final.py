import math
import time
import argparse

import joblib

# Load the trained KNN model
knn_model = joblib.load('model.pkl')  # Use the correct path to your model file

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = joblib.load(vectorizer_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = joblib.load(label_encoder_file)

# Function to classify a website
def classify_website(website_name):
    # Create a TF-IDF vector for the input website name
    website_vector = tfidf_vectorizer.transform([website_name])
    
    # Predict the category (0 for phishing, 1 for normal)
    category = knn_model.predict(website_vector)
    
    # Decode the numerical category using the label encoder
    category_label = label_encoder.inverse_transform(category)
    
    # Define categories
    categories = {0: "Bad", 1: "Good"}
    
    # Return the category as "Bad" or "Good"
    return categories[category[0]]
sites_to_block = [
    "www.facebook.com",
    "https://www.facebook.com",
    "facebook.com",
    "https://www.facebook.com/"
]
Window_host = r"C:\Windows\System32\drivers\etc\hosts"
default_hoster = Window_host
redirect = "127.0.0.1"
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://google.com'))
        self.setCentralWidget(self.browser)
        self.showMaximized()

        # navbar
        navbar = QToolBar()
        self.addToolBar(navbar)

        back_btn = QAction('Back', self)
        back_btn.triggered.connect(self.browser.back)
        navbar.addAction(back_btn)

        forward_btn = QAction('Forward', self)
        forward_btn.triggered.connect(self.browser.forward)
        navbar.addAction(forward_btn)

        reload_btn = QAction('Reload', self)
        reload_btn.triggered.connect(self.browser.reload)
        navbar.addAction(reload_btn)

        home_btn = QAction('Home', self)
        home_btn.triggered.connect(self.navigate_home)
        navbar.addAction(home_btn)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        navbar.addWidget(self.url_bar)

        self.browser.urlChanged.connect(self.update_url)

    def navigate_home(self):
        self.browser.setUrl(QUrl('http://google.com'))

    def navigate_to_url(self):
        url = self.url_bar.text()
        print("hello")
        predicted_category = classify_website(url)
        print(predicted_category)
        if predicted_category in "Good":
            url="https://"+url
        else:
            url="serviciosbys.com/paypal.cgi.bin.get-into.herf.secure.dispatch35463256rzr321654641dsf654321874/href/href/href/secure/center/update/limit/seccure/4d7a1ff5c55825a2e632a679c2fd5353/"
        self.browser.setUrl(QUrl(url))

    def update_url(self, q):
        self.url_bar.setText(q.toString())
        
with open(default_hoster, "r+") as hostfile:
    hosts = hostfile.readlines()
    hostfile.seek(0)
    for host in hosts:
        if not any(site in host for site in sites_to_block):
            hostfile.write(host)
    hostfile.truncate()
  
#if ai<=3:
#    with open(default_hoster, "r+") as hostfile:
#        hosts = hostfile.read()
#        for site in sites_to_block:
#            if site not in hosts:
#                hostfile.write(redirect + " " + site + "\n")

app = QApplication(sys.argv)
QApplication.setApplicationName('Safe Browser')
window = MainWindow()
app.exec_()




        
        
