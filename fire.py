import cv2
from tensorflow import keras
import numpy as np

model = keras.models.load_model("my_model.h5")

class_labels = ["Fire", "No Fire"]

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size
    img = cv2.resize(frame, (224, 224))  # 224x224 for most CNNs

    # Normalize and expand dimensions for prediction
    x = np.expand_dims(img, axis=0) / 255.0

    # Make prediction
    preds = model.predict(x)
    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]

    # Label to display
    label = f"{class_labels[class_index]}: {confidence*100:.2f}%"

    # Color: red if Fire, green if No Fire
    color = (0, 0, 255) if class_index == 0 else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    # Show the video frame
    cv2.imshow("Fire Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()