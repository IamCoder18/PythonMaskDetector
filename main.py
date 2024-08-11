from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(900),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.2),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

test_dataset = core.Dataset('Test/', transform=custom_transforms)  # L1
train_dataset = core.Dataset('Train/', transform=custom_transforms)  # L2
loader = core.DataLoader(test_dataset, batch_size=2, shuffle=True)  # L3
model = core.Model(['Mask', 'No Mask'])  # L4
losses = model.fit(loader, train_dataset, epochs=25, lr_step_size=5, learning_rate=0.001,
                   verbose=True)  # L5

model.save('model_weights.pth')
model = core.Model.load('model_weights.pth', ['Mask', 'No Mask'])

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite('image.png', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

image = utils.read_image('image.png')
predictions = model.predict(image)
labels, boxes, scores = predictions
show_labeled_image(image, boxes, labels)