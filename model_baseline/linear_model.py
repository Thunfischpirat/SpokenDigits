from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

classifier = make_pipeline(
    StandardScaler(),
    SGDClassifier(penalty="elasticnet", loss="modified_huber", n_jobs=-1),
)

if __name__ == "__main__":
    from model_baseline.mel_spectogram import create_features

    num_mels = 13
    num_frames = 10
    split = "TRAIN"

    train_features, train_labels = create_features(split, num_mels, num_frames)
    classifier.fit(train_features, train_labels)
    print(f"Accuracy on training set: {classifier.score(train_features, train_labels):.3f}\n")

    split = "DEV"
    dev_features, dev_labels = create_features(split, num_mels, num_frames)
    print(f"Accuracy on validation set: {classifier.score(dev_features, dev_labels):.3f}\n")

    split = "TEST"
    test_features, test_labels = create_features(split, num_mels, num_frames)
    print(f"Accuracy on test set: {classifier.score(test_features, test_labels):.3f}\n")
