import wandb
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_square_error

wandb.init(project="mlops-zoomcamp-wandb", name="experiment-1")

# ## Load the dataset
X, y = load_iris(return_X_y=True)
label_names = ["Setosa", "Versicolor", "Virginica"]

# ## Training model and Experimental Tracking
params = {"C":0.1, "random_state":42}
wandb.config = params

model = LogisticRegression(**params).fit(X, y)
y_pred = model.predict(X)
y_probas = model.predict_proba(X)

# Log your metrics to Weights & Biases using ``wandb.log``
wandb.log({
    "accuracy": accuracy_score(y, y_pred),
    "mean_square_error": mean_square_error(y, y_pred)
})
# ## Visualize and compare plots
# 
# We plot the **ROC curves** to compare the true positive rate vs the false positive rate for the different models.

wandb.sklearn.plot_roc(y, y_probas, labels=label_names)

# We plot the **confusion matrix** to compare the performance of the different models.

wandb.sklearn.plot_confusion_matrix(y, y_pred, labels=label_names)

# ## Logging Model to Weights & Biases

# save your model
with open("Logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)

# Log your model as versioned file to weights & Biases Artifact
artifact = wandb.Artifact("Logistic_regression", type="model")
artifact.add_file("Logistic_regression.pkl")
wandb.log_artifact(artifact)

# ## Finish the experiment
wandb.finish()

