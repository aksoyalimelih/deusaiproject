from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

app = Flask(__name__)

# Upload ve output dizinleri
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
SEGMENTED_FOLDER = os.path.join(OUTPUT_FOLDER, 'segmented')
AUGMENTED_FOLDER = os.path.join(OUTPUT_FOLDER, 'augmented')

# Klasörlerin oluşturulması
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Segmentasyon ve artırma işlemleri
            segmented_images = segment_image(filepath, filename)
            augmented_images = augment_image(filepath, filename)

            return render_template('result.html', segmented_images=segmented_images, augmented_images=augmented_images)
    return render_template('upload.html')

def resize_image(image, target_size=(640, 480)):
    """Resmi verilen hedef boyuta yeniden boyutlandır."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def segment_image(filepath, filename):
    # Resmi yükle
    image = cv2.imread(filepath)
    # Resmi yeniden boyutlandır
    image = resize_image(image)

    # GrabCut için başlangıç maskesi oluştur
    mask = np.zeros(image.shape[:2], np.uint8)

    # Arka plan ve nesne modelleri
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Segmentasyonu başlatmak için daha iyi bir dikdörtgen belirleme
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

    # GrabCut algoritmasını uygulayın
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)

    # Maskeyi işleyin
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image * mask2[:, :, np.newaxis]

    # Morfolojik işlemler uygulayın
    kernel = np.ones((5, 5), np.uint8)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

    # Segmentasyon aşamalarının kaydedilmesi
    step_images = [
        ("Segmentasyon Sonucu (GrabCut)", segmented)
    ]

    segmented_filenames = []
    for i, (desc, img) in enumerate(step_images):
        step_filename = f'step_{i+1}_{filename}.png'  # Uzantıyı ekleyin
        step_path = os.path.join(SEGMENTED_FOLDER, step_filename)
        cv2.imwrite(step_path, img)
        segmented_filenames.append((desc, step_filename))

    return segmented_filenames

def augment_image(filepath, filename):
    # Resmi yükle
    image = cv2.imread(filepath)
    # Resmi yeniden boyutlandır
    image = resize_image(image)

    # Augmentasyon işlemleri
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Yatay çevirme
        iaa.Flipud(0.5),  # Dikey çevirme
        iaa.Affine(rotate=(-20, 20)),  # Dönme
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Bulanıklaştırma
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Gürültü ekleme
        iaa.Multiply((0.8, 1.2)),  # Parlaklık değişikliği
        iaa.Crop(percent=(0, 0.1)),  # Kırpma
    ])
    
    # Resmi augment et
    images_aug = seq(images=[image]*5)  # 5 farklı artırılmış resim

    augmented_filenames = []
    for i, aug_img in enumerate(images_aug):
        augmented_filename = f'augmented_{i}_{filename}.png'  # Uzantıyı ekleyin
        augmented_path = os.path.join(AUGMENTED_FOLDER, augmented_filename)
        cv2.imwrite(augmented_path, aug_img)
        augmented_filenames.append((f"Augmented {i + 1}", augmented_filename))

    return augmented_filenames

if __name__ == '__main__':
    app.run(debug=True)