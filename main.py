import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint

# Load the webcam
cap = cv2.VideoCapture(0)

# Create the model
model = Sequential()

# Add the convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(320, 240, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add the fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        callbacks=[ModelCheckpoint('weights.h5f', monitor='val_accuracy', save_best_only=True)])

# Load the best weights
model.load_weights('weights.h5f')

# Start the prediction loop
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)

    # Get the prediction
    prediction = model.predict(frame)

    # Print the prediction
    print(prediction)

    # Show the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the key is ESC, stop the loop
    if key == 27:
        break

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()