import base64
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path='your_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_and_preprocess_image(image_data, target_size=(150, 150)):
    try:
        img = tf.image.decode_jpeg(image_data, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0 
        img = tf.expand_dims(img, axis=0) 
        return img
    except Exception as e:
        raise ValueError("Error processing image: {}".format(str(e)))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image data found'})

        base64_image = request.json['image']
        image_data = base64.b64decode(base64_image)

        input_img = load_and_preprocess_image(image_data)

        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({'prediction': output.tolist()})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        # Log the error
        app.logger.error("Prediction error: %s", str(e))
        return jsonify({'error': 'An error occurred during prediction'})



