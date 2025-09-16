# app/routes.py

import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- BAGIAN INI SANGAT PENTING ---
# Menambahkan folder root (DEEP-STEGA) ke dalam path Python
# Ini memungkinkan kita mengimpor dari folder `models` dan `utils`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ------------------------------------

# --- PERBAIKAN IMPOR ---
# Impor langsung dari file backend karena mereka berada di folder `app` yang sama
from encoder_backend import process_encoding
from decoder_backend import process_decoding
# -----------------------

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Konfigurasi folder upload (relatif terhadap folder app)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FOLDER)

# Pastikan folder ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    if 'cover_video' not in request.files or 'secret_images' not in request.files:
        flash('Form tidak lengkap!')
        return redirect(url_for('index'))

    cover_video = request.files['cover_video']
    secret_images = request.files.getlist('secret_images')
    frame_position = int(request.form['frame_position'])

    if cover_video.filename == '' or not secret_images:
        flash('File video atau gambar rahasia belum dipilih!')
        return redirect(url_for('index'))

    # Simpan file yang di-upload
    video_filename = secure_filename(cover_video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    cover_video.save(video_path)
    
    secret_paths = []
    for img in secret_images:
        img_filename = secure_filename(img.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        img.save(img_path)
        secret_paths.append(img_path)

    try:
        # Jalankan backend
        output_video_filename = process_encoding(video_path, secret_paths, frame_position)
        
        # Tampilkan hasil (pastikan nama file template benar: 'results.html')
        return render_template('results.html', 
                               mode='encode', 
                               stego_video_url=url_for('static', filename=f'results/{output_video_filename}'),
                               frame_position=frame_position)
    except Exception as e:
        flash(f"Terjadi error: {e}")
        return redirect(url_for('index'))

@app.route('/decode', methods=['POST'])
def decode():
    if 'stego_video' not in request.files:
        flash('File video stego belum dipilih!')
        return redirect(url_for('index'))
        
    stego_video = request.files['stego_video']
    frame_position = int(request.form['frame_position_decode'])
    num_secrets = int(request.form['num_secrets_decode'])

    # Simpan video stego
    video_filename = secure_filename(stego_video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    stego_video.save(video_path)

    try:
        # Jalankan backend
        revealed_image_filenames = process_decoding(video_path, frame_position, num_secrets)
        revealed_urls = [url_for('static', filename=f'results/{fname}') for fname in revealed_image_filenames]
        
        # Tampilkan hasil (pastikan nama file template benar: 'results.html')
        return render_template('results.html',
                               mode='decode',
                               revealed_images=revealed_urls)
    except Exception as e:
        flash(f"Terjadi error: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)