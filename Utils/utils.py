import gc
import numpy as np
import torch
import torch.nn as nn
from IPython.display import clear_output
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from tqdm import tqdm


def compute_loss(batch_pred, batch_true, device=torch.device('cuda:0')):
    return nn.CrossEntropyLoss()(batch_pred, batch_true.to(device))


def clip_batch(batch, size=500):
    texts = batch[1][:, :size]
    texts[:, -1] = 2
    return (batch[0], texts)


def get_all_y(model, data, predict_proba=False):
    y_true = []
    y_pred = []
    for batch in data:
        y_true.extend(list((batch[0]).cpu().detach().numpy()))
        if not predict_proba:
            y_pred.extend(list(model.predict(batch[1]).cpu().detach().numpy()))
        else:
            y_pred.extend(list(model(batch[1]).cpu().detach().numpy()))
    return np.array(y_true), np.array(y_pred)


def get_scores(model, data, print_scores=False, from_iter=False):
    if not from_iter:
        y_true = (data[0]).cpu().detach().numpy()
        y_pred = model.predict(data[1]).cpu().detach().numpy()
    else:
        y_true, y_pred = get_all_y(model, data)

    scores = [accuracy_score(y_true, y_pred),
              precision_score(y_true, y_pred),
              recall_score(y_true, y_pred)]

    if print_scores:
        print('Scores for {}:'.format(model.name))
        print('Accuracy: {}'.format(scores[0]))
        print('Precision: {}'.format(scores[1]))
        print('Recall: {}'.format(scores[2]))

    return np.array(scores)


def plot_roc_curve(model, data):
    y_true, y_pred = get_all_y(model, data, predict_proba=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1], pos_label=1)
    plt.figure(figsize=(15, 8))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid(c='g')
    plt.title('ROC-curve for {}'.format(model.name))
    auc_score = auc(fpr, tpr)
    print('Area under ROC-curve: {:1.4f}'.format(auc_score))
    plt.plot(fpr, tpr)
    plt.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), 'r')
    plt.show()


def train_model(model, opt, train_iter, val_iter, device=torch.device('cuda:0'), n_epochs=1):
    metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'grad_norm': []}

    step = 0
    for i in tqdm(range(n_epochs)):
        for batch in train_iter:
            batch = clip_batch(batch)
            step += 1
            model.train()

            loss_t = compute_loss(model(batch[1]), batch[0], device=device)
            loss_t.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10)

            opt.step()
            opt.zero_grad()

            gc.collect()

            if step % 100 == 99:
                metrics['grad_norm'].append(grad_norm)
                model.eval()
                metrics['train_loss'].append(loss_t.item())

                val_losses = []
                for val_batch in val_iter:
                    val_batch = clip_batch(val_batch)
                    val_losses.append(compute_loss(model(val_batch[1]), val_batch[0], device=device).item())
                metrics['val_loss'].append(np.mean(np.array(val_losses)))

                val_true, val_pred = get_all_y(model, val_iter)
                metrics['val_accuracy'].append(accuracy_score(val_true, val_pred))

                model.eval()
                clear_output(True)
                plt.figure(figsize=(16, 4))
                for i, (name, history) in enumerate(sorted(metrics.items())):
                    plt.subplot(1, len(metrics), i + 1)
                    plt.title(name)
                    plt.plot((np.arange(len(history)) + 1) * 100, history)
                    plt.grid()
                plt.show()
                print("Mean loss=%.3f" % np.mean(metrics['train_loss'][-5:], axis=0), flush=True)
