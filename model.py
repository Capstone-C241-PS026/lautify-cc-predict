import cv2
import numpy as np
import torch
from keras.models import load_model
import os
from google.cloud import storage
import datetime
import random
import string

class FishFreshnessModel:
    def __init__(self, detection_model_path, freshness_model_path, classification_model_path):
        self.storage_client = storage.Client()
        self.detection_model = self.load_torch_model(detection_model_path)
        self.freshness_model = self.load_keras_model(freshness_model_path)
        self.classification_model = self.load_keras_model(classification_model_path)

    def load_torch_model(self, local_path):
        model = torch.hub.load('yolov5', 'custom', path=local_path, source='local')
        return model

    def load_keras_model(self, local_path):
        model = load_model(local_path)
        return model

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        destination_blob_name= f'images-predict/{destination_blob_name}'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name, if_generation_match=blob.generation)
        print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}.")

    def preprocess_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    async def detect_eye_and_classify_freshness(self, image_bytes, output_bucket):
        np_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        results = self.detection_model(img)
        if len(results.xyxy[0]) == 0:
            raise ValueError("No fish eye detected in the image.")

        predictions = []
        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det

            cropped_eye = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            processed_eye = self.preprocess_image(cropped_eye)

            prediction = self.freshness_model.predict(processed_eye)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = prediction[0][predicted_class[0]]

            class_labels = ['Fresh', 'Not Fresh']
            freshness = class_labels[predicted_class[0]]

            predictions.append({"freshness": freshness, "confidence": int(float(confidence) * 100)})

            # Set bounding box color based on freshness
            if freshness == 'Fresh':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            # Draw bounding box and text
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(img, f"{freshness}", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        classify_img =self.preprocess_image(img)
        fish_prediction = self.classification_model.predict(classify_img)
        fish_class = np.argmax(fish_prediction, axis=1)

        fish_labels= ['Bawal Putih', 'Belato', 'Cakalang', 'Gembolo', 'Gole Gole', 'Kakap Merah', 'Kembung', 'Kerapu', 'Tenggiri', 'Tongkol']
        fish_kind = fish_labels[fish_class[0]]

        img_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
        img_path = os.path.join('temp', img_name)
        cv2.imwrite(img_path, img)

        print(f"Uploading annotated image to gs://{output_bucket}/{img_name}")
        self.upload_blob(output_bucket, img_path, img_name)

        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        result = {
            "image": f"https://storage.googleapis.com/lautify.appspot.com/images-predict/{img_name}?random={random_string}",
            "predictions": predictions,
            "fish_kind": fish_kind
        }
        return result
