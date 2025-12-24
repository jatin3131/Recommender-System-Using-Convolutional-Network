Recommender-System-Using-Convolutional-Network
advanced recommendation system for  an e-commerce platform that suggests  visually similar products using Convolutional Neural Networks (CNNs).(Image recognition)


Visual Product Recommendation System
Image-Based Recommendation using CNN (ResNet-50), PyTorch, and Flask
Table of Contents

Overview

Motivation

Features

System Architecture

Model Details

Similarity Measurement

Web Application

Project Structure

Installation & Setup

Usage

Limitations

Future Improvements

Conclusion

Overview

This project implements a visual product recommendation system for e-commerce platforms.
Instead of relying on text descriptions or user history, the system recommends products based on visual similarity between product images.

A deep learning model extracts meaningful visual features from product images, and similar products are identified using similarity metrics. The system is integrated into a web application using Flask.

Motivation

Traditional recommendation systems depend on:

Textual product descriptions

User interaction history

These approaches have limitations:

Poor or missing product descriptions

Cold-start problem for new products

Visual appearance not fully captured by text

This project focuses on computer vision–based recommendations, which are especially useful for fashion and lifestyle products.

Features

Image-based product recommendation

Deep feature extraction using CNN

Pretrained ResNet-50 model

Cosine similarity for matching products

Flask backend integration

Custom HTML and CSS frontend

Individual product pages with similar item suggestions

Modular and extensible design

System Architecture

Workflow:

Product images are loaded into the system

A CNN extracts deep feature vectors from each image

Feature vectors are stored for comparison

When a product is selected:

Its feature vector is compared with others

Most visually similar products are retrieved

Results are displayed on the web interface

Model Details
Neural Network Used

ResNet-50

Type: Residual Convolutional Neural Network

Framework: PyTorch

Pretrained on ImageNet dataset

Why ResNet-50?

Deep architecture with 50 layers

Uses residual (skip) connections

Prevents vanishing gradient problem

Excellent for feature extraction

The final classification layer is removed, and the network is used only to extract 2048-dimensional feature vectors from images.

Similarity Measurement

Cosine Similarity is used to compare image feature vectors

Measures angular similarity between vectors

High similarity score means visually similar products

Suitable for high-dimensional embeddings

Web Application

Backend built using Flask

Frontend built using HTML and CSS (no Bootstrap)

Flask handles:

Routing

Model inference

Passing recommendations to templates

Pages

Home Page: Product grid with square image cards

Product Page: Product details with similar product recommendations

Cart Page: Placeholder for future expansion




----------
Installation & Setup
Requirements

Python 3.9+

Flask

PyTorch

Torchvision

Pillow

NumPy


Steps

Clone the repository:

git clone https://github.com/your-username/visual-product-recommendation.git


Navigate to the project folder:

cd visual-product-recommendation


Install dependencies:

pip install -r requirements.txt


Add product images to:

static/images/


Run the application:

python app.py


Open in browser:

http://127.0.0.1:5000

------
Usage

Open the home page to view products

Click on any product image

View product details

Scroll down to see visually similar products

Limitations

No user behavior or personalization

Performance depends on image quality

Limited dataset size

Cropped-region recommendation requires further optimization

Future Improvements

Region-based image selection (crop and recommend)

Hybrid recommendations (image + user behavior)

Database integration (MySQL / PostgreSQL)

User authentication and cart system

Fine-tuning CNN on domain-specific data

Cloud deployment

Conclusion

This project demonstrates how deep learning and computer vision can be used to build an intelligent, image-based recommendation system. By leveraging pretrained CNN models and similarity metrics, it provides a strong foundation for modern e-commerce applications where visual appearance plays a key role.
