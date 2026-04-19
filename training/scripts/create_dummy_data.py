# create_dummy_data.py
from PIL import Image
import os

base = r"D:\object-detection(project)\dataset"
os.makedirs(os.path.join(base, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(base, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(base, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(base, "labels", "val"), exist_ok=True)

im = Image.new('RGB', (640, 480), (128, 128, 128))
im.save(r"D:\object-detection(project)\dataset\images\train\test_0001.jpg")
im.save(r"D:\object-detection(project)\dataset\images\val\test_0001.jpg")
open(r"D:\object-detection(project)\dataset\labels\train\test_0001.txt", 'w').close()
open(r"D:\object-detection(project)\dataset\labels\val\test_0001.txt", 'w').close()
print("Dummy images created.")
