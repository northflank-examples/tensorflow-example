import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List
import base64
import io
from PIL import Image
import numpy as np

tf.config.optimizer.set_jit(True)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU acceleration enabled: {len(physical_devices)} GPU(s) found")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled")
else:
    print("No GPU found, using CPU")

app = FastAPI(title="TensorFlow Fashion MNIST API", version="1.0.0")

model = None
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class TrainRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64

class PredictRequest(BaseModel):
    image_base64: str

class TrainResponse(BaseModel):
    message: str
    epochs_completed: int
    final_accuracy: float
    final_loss: float

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[float]

def create_model():
    with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(10, dtype='float32')
        ])
    return model

def initialize_model():
    global model
    
    model = create_model()
    
    if os.path.exists("model.keras"):
        try:
            model = keras.models.load_model("model.keras")
            print("Loaded existing model from model.keras")
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model.")
            model = create_model()

def get_data(batch_size: int = 64):
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, (x_test, y_test)

def preprocess_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        if image.size != (28, 28):
            image = image.resize((28, 28))
        
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/")
async def root():
    gpu_info = {
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        "gpu_count": len(tf.config.list_physical_devices('GPU')),
        "mixed_precision": tf.keras.mixed_precision.global_policy().name
    }
    return {
        "message": "TensorFlow Fashion MNIST API is running", 
        "model_loaded": model is not None,
        "gpu_info": gpu_info
    }

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        print(f"Starting training with {request.epochs} epochs...")
        
        train_dataset, test_dataset, (x_test, y_test) = get_data(request.batch_size)
        
        optimizer = keras.optimizers.SGD(learning_rate=request.learning_rate)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        class PrintProgress(keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if batch % 100 == 0:
                    print(f"batch {batch}: loss: {logs['loss']:>7f}")
        
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            history = model.fit(
                train_dataset,
                epochs=request.epochs,
                validation_data=test_dataset,
                callbacks=[PrintProgress()],
                verbose=1
            )
        
        final_loss = history.history['val_loss'][-1]
        final_accuracy = history.history['val_accuracy'][-1]
        
        model.save("model.keras")
        print("Model saved to model.keras")
        
        return TrainResponse(
            message="Training completed successfully",
            epochs_completed=request.epochs,
            final_accuracy=float(final_accuracy),
            final_loss=float(final_loss)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        image_array = preprocess_image(request.image_base64)
        
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            logits = model(image_array, training=False)
            probabilities = tf.nn.softmax(logits).numpy()[0]
            predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]
        
        confidence = float(probabilities[predicted_class_idx])
        
        return PredictResponse(
            predicted_class=classes[predicted_class_idx],
            confidence=confidence,
            probabilities=probabilities.tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    global model
    
    if model is None:
        return {"status": "Model not initialized"}
    
    total_params = model.count_params()
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpu_devices) > 0
    device_info = f"GPU ({len(gpu_devices)} devices)" if gpu_available else "CPU"
    
    return {
        "status": "Model initialized",
        "device": device_info,
        "gpu_devices": [str(device) for device in gpu_devices],
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
        "total_parameters": int(total_params),
        "trainable_parameters": int(total_params),
        "model_file_exists": os.path.exists("model.keras"),
        "classes": classes,
        "tensorflow_version": tf.__version__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 