import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('kidney_model_best.h5')

# Label mapping
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('designer.ui', self)

        # Connect button click events to functions
        self.select_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.classify_image)

        # Set initial text for result_label
        self.result_label.setText("")

        # Initialize img_array attribute
        self.img_array = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.png *.jpeg *.webp)")
        if file_path:
            image = Image.open(file_path)
            image = image.resize((400, 400))  # Resize to your desired dimensions
            image = image.convert("RGB")  # Convert to RGB mode
            q_image = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas.setPixmap(pixmap)
            
            # Set the img_array attribute
            self.img_array = np.array(image)

            self.detect_button.setEnabled(True)  # Enable the classify button


    def classify_image(self):
        if self.img_array is not None:
            try:
                image = Image.fromarray(self.img_array)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(image)

                # Extract predicted class index and probability
                predicted_class_index = np.argmax(predictions)
                predicted_class = labels[predicted_class_index]
                predicted_percentage = predictions[0][predicted_class_index] * 100

                # Update result label
                result_text = f"Predicted Class: {predicted_class}\nPredicted Percentage: {predicted_percentage:.2f}%"
                self.result_label.setText(result_text)

            except Exception as e:
                QMessageBox.warning(self, "Error", "Error occurred during classification.")
                print("Error:", e)
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

def main():
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
