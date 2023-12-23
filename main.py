import base64
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='your_model.tflite')
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_and_preprocess_image(image_data, target_size=(150, 150)):
    img = tf.image.decode_jpeg(image_data, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image data found'})

        base64_image = request.json['image']
        image_data = base64.b64decode(base64_image)

        # Preprocess the image using the provided function
        input_img = load_and_preprocess_image(image_data)

        # Set the image data as input to the model
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()

        # Get the model output
        output = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({'prediction': output.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

