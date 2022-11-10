import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model.eval()

true_labels = []
pred_labels = []

for i, sample_batch in enumerate(test_loader):
    sequences = sample_batch['sequence']
    true_batch = sample_batch['label']

    outputs = model(sequences)
    _, pred_batch = torch.max(outputs, 1)

    true_labels.append(true_batch.detach().numpy())
    pred_labels.append(pred_batch.detach().numpy())

true_labels = np.concatenate(true_labels).squeeze()
pred_labels = np.concatenate(pred_labels)

acc = (true_labels == pred_labels).sum() / len(true_labels)
cm = confusion_matrix(true_labels, pred_labels)
cr = classification_report(true_labels, pred_labels)
print(acc)
print(cm)
print(cr)
