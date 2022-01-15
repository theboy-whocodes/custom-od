from cProfile import label
from utils.bbox_regressor import ObjecDetector
from utils.custom_tensor_dataset import CustomTensorDataset
from utils import config
from tqdm import tqdm
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os


print(f"[INFO] Load dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
    rows = open(csvPath).read().strip().split("\n")

    for row in rows:
        row = row.split(",")
        (filename, startX, startY, endX, endY, lebel) = row

        imagePath = os.path.sep.join([config.IMAGES_PATH, lebel, filename])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]

        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)


data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

le = LabelEncoder()
labels = le.fit_transform(labels)

split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size=0.2, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

(trainImages, testImages) = torch.tensor(trainImages), torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels), torch.tensor(testLabels)
(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes), torch.tensor(testBBoxes)

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

trainDS = CustomTensorDataset(
    (trainImages, trainLabels, trainBBoxes), transforms=transforms)
testDS = CustomTensorDataset(
    (testImages, testLabels, testBBoxes), transforms=transforms)
print(f"[INFO] Total training samples: {len(trainDS)}")
print(f"[INFO] Total test samples: {len(testDS)}")

trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(testDS) // config.BATCH_SIZE

trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE, shuffle=True,
                         num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
                        num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)

print(f"[INFO] Saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.writable("\n".join(testPaths))
f.close()

resnet = resnet50(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

objectDetector = ObjecDetector(resnet, len(le.classes_))
objectDetector = objectDetector.to(config.DEVICE)

classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
print(objectDetector)

H = {"total_train_loss": [], "total_val_loss": [],
     "train_class_acc": [], "val_class_acc": []}

print(f"[INFO] Training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    objectDetector.train()

    totalTrainLoss = 0
    totalValLoss = 0

    trainCorrect = 0
    valCorrect = 0

    for (images, labels, bboxes) in trainLoader:
        (images, labels, bboxes) = (images.to(config.DEVICE),
                                    labels.to(config.DEVICE), bboxes.to(config.DEVICE))

        predictions = objectDetector(images)
        bboxLoss = bboxLossFunc(predictions[0], bboxes)
        classLoss = classLossFunc(predictions[1], labels)
        totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)

        opt.zero_grad()
        totalLoss.backward()
        opt.step()

        totalTrainLoss += totalLoss
        trainCorrect += (predictions[1].argmax(1)
                         == labels).type(torch.float).sum().item()

    with torch.no_grad():
        objectDetector.eval()

        for (images, labels, bboxes) in testLoader:
            (images, labels, bboxes) = (images.to(config.DEVICE),
                                        labels.to(config.DEVICE), bboxes.to(config.DEVICE))

            predictions = objectDetector(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)
            totalValLoss += totalLoss
            valCorrect += (predictions[1].argmax(1)
                           == labels).type(torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(testDS)

        H['total_train_loss'].append(avgTrainLoss.cpu().detach().numpy())
        H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)

        print(f"[INFO] EPROCH: {e+1}/{config.NUM_EPOCHS}")
        print("[INFO] Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("[INFO] Val Loss: {:.6f}, Val Accuracy: {:.4f}".format(
            avgValLoss, valCorrect))
endTime = time.time()
print("[INFO] Total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
print("[INFO] Saving object detector modl...")
torch.save(objectDetector, config.MODEL_PATH)

print("[INFO] Saving label encoder...")
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["total_val_loss"], label="total_val_loss")
plt.plot(H["train_class_acc"], label="train_class_acc")
plt.plot(H["val_class_acc"], label="val_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)
