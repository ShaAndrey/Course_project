import datetime
import gc
import random
import time

import torch
import torch.nn as nn
from IPython.display import clear_output

import numpy as np
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


def get_scores(model, data, print_scores=False, from_iter=False, get_function=get_all_y):
    if not from_iter:
        y_true = (data[0]).cpu().detach().numpy()
        y_pred = model.predict(data[1]).cpu().detach().numpy()
    else:
        y_true, y_pred = get_function(model, data)

    scores = [accuracy_score(y_true, y_pred),
              precision_score(y_true, y_pred),
              recall_score(y_true, y_pred)]

    if print_scores:
        print('Scores for {}:'.format(model.name))
        print('Accuracy: {}'.format(scores[0]))
        print('Precision: {}'.format(scores[1]))
        print('Recall: {}'.format(scores[2]))

    return np.array(scores)


def plot_roc_curve(model, data, get_function=get_all_y):
    y_true, y_pred = get_function(model, data, predict_proba=True)

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


def predicted_labels(preds):
    return np.argmax(preds, axis=1).flatten()


def flat_accuracy(preds, labels):
    pred_flat = predicted_labels(preds)
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_bert(model, optimizer, train_dataloader, val_dataloader, device=torch.device('cuda:0'), n_epochs=1):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    total_t0 = time.time()

    for epoch_i in range(0, n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss, logits = output.loss, output.logits

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        val_accuracy, avg_val_loss, validation_time = eval_bert(model, val_dataloader)

        print("  Accuracy: {0:.2f}".format(val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_accuracy'].append(val_accuracy)

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return metrics


def eval_bert(model, dataloader, device=torch.device('cuda:0')):
    model.eval()

    t0 = time.time()

    total_eval_accuracy = 0
    total_eval_loss = 0
    total_length = 0

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss, logits = output.loss, output.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids) * len(label_ids.flatten())
        total_length += len(label_ids.flatten())

    accuracy = total_eval_accuracy / total_length
    avg_loss = total_eval_loss / len(dataloader)

    eval_time = format_time(time.time() - t0)

    return accuracy, avg_loss, eval_time


def get_all_y_for_bert(model, data, predict_proba=False):
    y_true = []
    y_pred = []
    for batch in data:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        y_true.extend(list((b_labels).cpu().detach().numpy()))

        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss, logits = output.loss, output.logits
        logits = logits.detach().cpu().numpy()
        if not predict_proba:
            y_pred.extend(list(predicted_labels(logits)))
        else:
            y_pred.extend(logits)
    return np.array(y_true), np.array(y_pred)