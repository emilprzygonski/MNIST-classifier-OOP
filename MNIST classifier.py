from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
 
class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image):
        pass
 
class CNNClassifier(DigitClassificationInterface):
    def __init__(self):
        self.model = self._build_model()
 
    def _build_model(self) -> tf.keras.Sequential:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
 
    def predict(self, image: np.ndarray)-> int:
        image = image.reshape((1, 28, 28, 1))
        prediction = self.model.predict(image)[0]
        return np.argmax(prediction)
 
class RFClassifier(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
 
    def predict(self, image: np.ndarray)-> int:
        image = image.reshape((1, -1))
        prediction = self.model.predict(image)
        return prediction[0]
 
class RandomClassifier(DigitClassificationInterface):
    def predict(self, image: np.ndarray)-> int:
        return np.random.randint(0, 10)
 
class DigitClassifier:
    def __init__(self, algorithm: str):
        if algorithm == 'cnn':
            self.model = CNNClassifier()
        elif algorithm == 'rf':
            self.model = RFClassifier()
        elif algorithm == 'rand':
            self.model = RandomClassifier()
        else:
            raise ValueError('Invalid algorithm name.')
    
    def train(self):
        raise NotImplementedError("Method not implemented yet.")

    def predict(self, image: np.ndarray)-> int:
        image = image.astype(np.float32) / 255.0
        return self.model.predict(image)