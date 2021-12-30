from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics  import classification_report
from preprocessing import SimplePreprocessor #our own created class
from  datasets import SimpleDatasetLoader #our own datasetloader
from imutils import paths

DATASET_PATH = 'datasets/animals'
K = 3 #for KNN value

#1. Prepare Data
print("Loading images....")
image_paths = list(paths.list_images(DATASET_PATH))
sp = SimplePreprocessor(32,32) #load from dataset and reshape to the data matrix
dataloader = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = dataloader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0],3072)) #because data.shape[0] is 3000 (all images), column is 32x32x3 = 30472

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1000.0)))

#2. Encode Label strings as numbers. For ex: dog - 0, cat-1 an so on.
le = LabelEncoder()
labels = le.fit_transform(labels)

#split data into train and test
									#25% for test and 75% for training,
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

# 3. Train and test model.
model = KNeighborsClassifier(n_neighbors=K,n_jobs=5)
model.fit(trainX,trainY)
print(classification_report(testY, model.predict(testX),target_names = le.classes_))
