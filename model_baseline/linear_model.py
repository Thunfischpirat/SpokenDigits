from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

classifier = make_pipeline(
    StandardScaler(),
    SGDClassifier(penalty="elasticnet", loss="modified_huber", n_jobs=-1),
)

if __name__ == "__main__":
    from model_baseline.data_loading import create_features

    num_mels = 13
    num_frames = 10
    split = "TRAIN"
    print("----------------------------------TRAIN-SET----------------------------------------")
    train_features, train_labels = create_features(split, num_mels, num_frames)
    classifier.fit(train_features, train_labels)
    train_preds = classifier.predict(train_features)
    print(f"Confusion matrix:\n{confusion_matrix(train_labels, train_preds)}\n")
    print(f"Classification Report:\n{classification_report(train_labels, train_preds)}\n")
    print("----------------------------------DEV-SET-----------------------------------------")
    split = "DEV"
    dev_features, dev_labels = create_features(split, num_mels, num_frames)
    dev_preds = classifier.predict(dev_features)
    print(f"Confusion matrix:\n{confusion_matrix(dev_labels, dev_preds)}\n")
    print(f"Classification Report:\n{classification_report(dev_labels, dev_preds)}\n")
    print("---------------------------------TEST_SET------------------------------------------")
    split = "TEST"
    test_features, test_labels = create_features(split, num_mels, num_frames)
    test_preds = classifier.predict(test_features)
    print(f"Confusion matrix:\n{confusion_matrix(test_labels, test_preds)}\n")
    print(f"Classification Report:\n{classification_report(test_labels, test_preds)}\n")
