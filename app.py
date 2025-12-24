from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import os, base64
from io import BytesIO
from PIL import Image
from model.feature_extractor import extract_features

app = Flask(__name__)
app.secret_key = "replace_with_a_secure_key"

# ---------- Load features ----------
with open("database/features.pkl","rb") as f:
    features_dict = pickle.load(f)

filenames = list(features_dict.keys())
feature_matrix = np.array([features_dict[f] for f in filenames])

# ---------- Similarity Function ----------
def get_top_k_similar(selected_image, k=5):
    if selected_image not in features_dict: 
        return []
    query_vec = features_dict[selected_image].reshape(1,-1)
    similarities = cosine_similarity(query_vec, feature_matrix)[0]
    top_indices = similarities.argsort()[::-1]
    top_indices = [i for i in top_indices if filenames[i] != selected_image][:k]
    return [filenames[i] for i in top_indices]

# ---------- Context Processor ----------
@app.context_processor
def inject_now():
    return {'current_year': datetime.datetime.now().year}

# ---------- Cart Helpers ----------
def ensure_cart():
    if 'cart' not in session:
        session['cart'] = []

# ---------- Routes ----------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', images=filenames)

@app.route('/product/<product_image>')
def product_page(product_image):
    if product_image not in filenames:
        return "Product not found", 404

    product = {
        "image": product_image,
        "name": product_image.split('.')[0].replace('_',' '),
        "price": 499,
        "category": "Clothing"
    }

    top5 = get_top_k_similar(product_image, k=5)

    return render_template('product.html', product=product, similar=top5)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected = request.form.get('selected_image')
    top5 = get_top_k_similar(selected, k=5)
    return render_template('index.html', images=filenames, top5=top5, selected_image=selected)

@app.route('/cart')
def cart_view():
    ensure_cart()
    return render_template('cart.html', cart=session['cart'])

@app.route('/cart/add', methods=['POST'])
def cart_add():
    ensure_cart()
    prod = request.form.get('product')
    qty = int(request.form.get('qty',1))
    item = {"image": prod, "name": prod.split('.')[0].replace('_',' '), "price": 499, "qty": qty}
    session['cart'].append(item)
    session.modified = True
    session['cart_count'] = sum(i['qty'] for i in session['cart'])
    return redirect(request.referrer or url_for('index'))

@app.route('/cart/remove', methods=['POST'])
def cart_remove():
    ensure_cart()
    prod = request.form.get('product')
    session['cart'] = [i for i in session['cart'] if i['image'] != prod]
    session.modified = True
    session['cart_count'] = sum(i['qty'] for i in session['cart'])
    return redirect(url_for('cart_view'))

# ---------- Crop & Recommend Route ----------
@app.route('/crop_upload', methods=['POST'])
def crop_upload():
    data = request.get_json()
    img_data = data['image']

    # Remove "data:image/jpeg;base64," prefix
    if "," in img_data:
        img_data = img_data.split(",")[1]

    try:
        img = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
    except:
        return jsonify({"error":"Invalid image"}), 400

    # Save temporary cropped image
    import os
    os.makedirs("static/temp", exist_ok=True)
    temp_path = "static/temp/cropped.jpg"
    img.save(temp_path)

    # Extract features
    cropped_feat = extract_features(temp_path)

    # Compare with all products
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = []
    for fname, feat in features_dict.items():
        sim = cosine_similarity(cropped_feat.reshape(1,-1), feat.reshape(1,-1))[0][0]
        similarities.append((fname, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = [fname for fname, _ in similarities if fname != os.path.basename(temp_path)][:6]

  

    return jsonify({"similar": top_similar})
# ---------- Run Server ----------
if __name__ == "__main__":
    app.run(debug=True)
