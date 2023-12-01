from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import shutil

errors = 200
total_deleted = 0

def preprocess_images(file_paths):
    processed_images = []
    for picture in file_paths:
        image = Image.open(picture).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        processed_images.append(normalized_image_array)
    return np.array(processed_images)

def Predict_If_Car(images, file_paths, model, class_names):
    global total_deleted
    deleted = 0
    data = np.array(images)
    predictions = model.predict(data)
    
    for i, prediction in enumerate(predictions):
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = f"{round((prediction[index])*100)}%"
        
        if index != 0:
            os.remove(file_paths[i])
            deleted += 1
            print(f"{file_paths[i]} is {confidence_score} a {class_name[2:]} and has been deleted")
            continue
        elif index == 0 and confidence_score != "100%":
            os.remove(file_paths[i])
            deleted += 1
            print(f"{file_paths[i]} might not be a {class_name[2:]} and has been deleted")
            continue
        else:
            print(f"{file_paths[i]} is {confidence_score} a {class_name[2:]}")
            continue
            
    total_deleted += deleted
    return total_deleted


model = load_model(r"C:\Users\Maste\Desktop\AI\converted_keras (2)\keras_model.h5", compile=False)
class_names = open(r"C:\Users\Maste\Desktop\AI\converted_keras (2)\labels.txt", "r").readlines()


deleted = 0
total_files = 0
BrandFile = rf"C:\Users\Maste\Desktop\AI\Cars\Porsche"


for CarFile in os.listdir(BrandFile):
    
    if len(CarFile) == None:
        continue
        
    file_paths = [os.path.join(BrandFile, CarFile, picture) for picture in os.listdir(os.path.join(BrandFile, CarFile))]
    for file_path in file_paths:
        
        if (file_path[-4:].lower() != ".jpg") and (file_path[-4:].lower() != ".png"):
            print(f"{file_path} deleted for not being a jpg or png")
            file_paths.remove(file_path)
            deleted += 1
            continue
        
        if (file_path[-4:].lower() == ".jpg") or (file_path[-4:].lower() == ".png"):
            total_files += 1
        
    
        if len(file_path) > 250:
            old_file_path = file_path
            new_file_path = f"{BrandFile}\{CarFile}\download({errors}).jpg"
            print(f"{old_file_path} is greater than character limit and has been changed to {new_file_path}")
            file_paths.remove(old_file_path)
            os.rename(old_file_path, new_file_path)
            errors += 1
            continue
        
    batch_size = 100
    for i in range(0, len(file_paths), batch_size):
        batch_file_paths = file_paths[i:i+batch_size]
        batch_images = preprocess_images(batch_file_paths)
        Predict_If_Car(batch_images, batch_file_paths, model, class_names)


total_deleted = Predict_If_Car(batch_images, file_paths, model, class_names)    
total_deleted += deleted
      
print(f"Finished Predictions, {total_deleted} files have been deleted, {total_files - total_deleted} have been kept")

shutil.move(BrandFile, r"C:\Users\Maste\Desktop\AI\Finished Cars")

