import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, recall_score, confusion_matrix
from basenet import ResNet50


class attribute_classifier():

    def __init__(self, device, dtype, modelpath=None, learning_rate=1e-4):
        # print(modelpath)
        self.model = ResNet50(n_classes=1, pretrained=True)
        self.model.require_all_grads()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.dtype = dtype
        self.iteration = 0
        self.best_acc = 0.0
        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            self.model.load_state_dict(A['model'])
            if (self.device == torch.device('cuda')):
                self.model.cuda()
            self.optimizer.load_state_dict(A['optim'])
            self.iteration = A['iteration']
            self.best_acc = A['best_acc']

    def forward(self, x):
        out, feature = self.model(x)
        return out, feature

    def save_model(self, path):
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict(), 'iteration': self.iteration,
                    'best_acc': self.best_acc}, path)

    def do_iteration(self, loader):
        """Train the model for one iteration"""

        self.model.train()

        images, targets = next(loader)

        self.model = self.model.to(device=self.device, dtype=self.dtype)

        images, targets = images.to(device=self.device, dtype=self.dtype), targets.to(device=self.device,
                                                                                      dtype=self.dtype)
        targets = targets[:, 0]

        self.optimizer.zero_grad()
        outputs, _ = self.forward(images)
        lossbce = torch.nn.BCEWithLogitsLoss()
        loss = lossbce(outputs.squeeze(), targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_scores(self, loader, labels_present=True):
        if (self.device == torch.device('cuda')):
            self.model.cuda()
        self.model.eval()  # set model to evaluation mode
        y_all = []
        scores_all = []
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)

                scores, _ = self.model(x)
                scores = torch.sigmoid(scores).squeeze()

                y_all.append(y.detach().cpu().numpy())
                scores_all.append(scores.detach().cpu().numpy())
            y_all = np.concatenate(y_all)
            pred_all = np.concatenate(scores_all)

        return y_all, pred_all

    def check_metrics(self, y_all, pred_all):

        # binary labels of the task attributes
        # e.g. similing
        # y_attr = [0, 1, 0, 0, 1, ... ]
        y_attr = torch.from_numpy(y_all[:, 0])

        # binary labels of the protected attributes
        # male
        # y_prot_attr = [0, 1, 1, 0, 1, ... ]
        y_prot_attr = torch.from_numpy(y_all[:, 1])

        # task attributes predictions
        # since it's a binary classification task the last FC outputs a single value
        # pred_all = [0.544433  , 0.5606193 , 0.54526544, ..., 0.5494616 ]
        # 0.5 is the standard threshold used for the binary task
        # pred = [True, True, True,  ..., True] True -> the image has the task attribute
        pred = torch.from_numpy(pred_all > 0.5)

        # accuracy of the binary prediction
        # if pred == True -> GT = 1
        # if pred == False -> GT = 0
        acc = torch.sum(pred == y_attr) / len(y_attr)

        # True Positive Rate (Sensitivity)
        # TPR = TP / (TP+FN)
        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_attr, pred).ravel()
        TPR = tp/(tp+fn)

        # min/max group accuracy ##############################

        # to select the indices where the protected attribute is 0
        # indices_prot_attr_0 = [    0,     2,     4, ..., 19862, 19864, 19865]
        indices_prot_attr_0 = np.where(y_prot_attr == 0)[0]
        indices_prot_attr_1 = np.where(y_prot_attr == 1)[0]

        # to select the task attribute GT where the protected attribute is 0
        y_task_attr_prot_attr_0 = y_attr[indices_prot_attr_0]
        y_task_attr_prot_attr_1 = y_attr[indices_prot_attr_1]

        # to select the task attribute prediction where the protected attribute is 0
        pred_task_attr_prot_attr_0 = pred[indices_prot_attr_0]
        pred_task_attr_prot_attr_1 = pred[indices_prot_attr_1]

        # accuracy computation
        acc_prot_attr_0 = torch.sum(pred_task_attr_prot_attr_0 == y_task_attr_prot_attr_0) / len(indices_prot_attr_0)
        acc_prot_attr_1 = torch.sum(pred_task_attr_prot_attr_1 == y_task_attr_prot_attr_1) / len(indices_prot_attr_1)

        # True Positive Rate min/max group
        tn_0, fp_0, fn_0, tp_0 = confusion_matrix(y_task_attr_prot_attr_0, pred_task_attr_prot_attr_0).ravel()
        TPR_prot_attr_0 = tp_0/(tp_0+fn_0)

        tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_task_attr_prot_attr_1, pred_task_attr_prot_attr_1).ravel()
        TPR_prot_attr_1 = tp_1/(tp_1+fn_1)

        # DEO computation
        # difference between the TPR on each subgroup.
        DEO = np.abs(TPR_prot_attr_0 - TPR_prot_attr_1)

        # False Positive Rate min/max group
        # FPR = FP / (FP+TN)
        FPR_prot_attr_0 = fp_0 / (fp_0 + tn_0)
        FPR_prot_attr_1 = fp_1 / (fp_1 + tn_1)

        # DEOOD computation
        # sum of the difference between the TPR and FPR on each subgroup
        DEODD = DEO + np.abs(FPR_prot_attr_0 - FPR_prot_attr_1)

        return acc, acc_prot_attr_0, acc_prot_attr_1, TPR, TPR_prot_attr_0, TPR_prot_attr_1, DEO, DEODD



