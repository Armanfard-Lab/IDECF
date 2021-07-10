import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np



def acc(y_true, y_pred, num_cluster):

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    w = np.zeros((num_cluster, num_cluster))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size


class IDECF(nn.Module):
    def __init__(self, AE, FCM_Net, data_loader, dataset_size, batch_size=500, pretraining_epoch = 200, max_iteration = 60, num_cluster = 10, m = 1.5, T1=10, landa =0.1, feature_size =1*28*28):
        super(IDECF, self).__init__()
        self.AE = AE
        self.FCM_Net = FCM_Net
        self.umean = torch.zeros([num_cluster,10])
        self.batch_size = batch_size
        self.pretraining_epoch = pretraining_epoch
        self.max_iteration = max_iteration
        self.num_cluster = num_cluster
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.m = m
        self.T1=T1
        self.landa = landa
        self.feature_size= feature_size

    def Kmeans_model_evaluation(self):
        self.AE.eval()
        datas = np.zeros([self.dataset_size, 10])
        label_true = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()

            _, u = self.AE(x.view(-1, self.feature_size))
            u = u.cpu()
            datas[ii * self.batch_size:(ii + 1) * self.batch_size, :] = u.data.numpy()
            label_true[ii * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)

        label_pred = kmeans.labels_
        ACC = acc(label_true, label_pred, self.num_cluster)
        print('ACC', ACC)
        return ACC

    def pretraining(self):
        self.AE.train()
        self.AE.cuda()
        for param in self.AE.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.AE.parameters())
        prev_ACC = 0
        for T in range(0,self.pretraining_epoch):
          print('Pretraining Iteration: ', T+1)
          for x, target in self.data_loader:
             optimizer.zero_grad()
             x = Variable(x).cuda()

             y, _ = self.AE(x.view(-1, self.feature_size))
             loss = nn.MSELoss()(x.view(-1, self.feature_size), y)
             loss.backward()
             optimizer.step()

          ACC = self.Kmeans_model_evaluation()
          if ACC > prev_ACC:
              prev_ACC = ACC
              with open('AE_MNIST_pretrain', 'wb') as f:
                    torch.save(self.AE, f)

        self.AE =torch.load('AE_MNIST_pretrain')
        return self.AE


    def c_means_cost(self,u, u_mean, p):
        return torch.sum(torch.mul(torch.mul(u - u_mean, u - u_mean), p))

    def update_cluster_centers(self):
        self.AE.eval()
        for param in self.AE.parameters():
            param.requires_grad = False
        for param in self.FCM_Net.parameters():
            param.requires_grad = False
        sum1 = torch.zeros([self.num_cluster]).cuda()
        sum2 = torch.zeros([self.num_cluster, 10]).cuda()
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x.view(-1, self.feature_size))
            p = self.FCM_Net(u)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                sum1[kk] = sum1[kk] + torch.sum(p[:, kk])
                sum2[kk, :] = sum2[kk, :] + torch.matmul(p[:, kk].T, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(sum2[kk, :], sum1[kk])
        return self.u_mean

    def FCM_Net_train(self):
        self.AE.eval()
        for param in self.AE.parameters():
            param.requires_grad = False
        for param in self.FCM_Net.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.FCM_Net.parameters())
        for TT in range(0, self.T1):
            for data, target in self.data_loader:
                x = Variable(data).cuda()
                _, u = self.AE(x.view(-1, self.feature_size))
                optimizer.zero_grad()
                p = self.FCM_Net(u)
                p = torch.pow(p, self.m)
                loss = self.c_means_cost(u.unsqueeze(1).repeat(1, 10, 1), self.u_mean, p.unsqueeze(2).repeat(1, 1, 10))
                loss.backward()
                optimizer.step()

        return self.FCM_Net

    def T_student(self, u, u_mean):
        q = torch.sum(torch.pow(1 + torch.pow(u - u_mean, 2), -1), dim=2)
        sum1 = torch.sum(q, dim=1)
        return torch.div(q, sum1.unsqueeze(1).repeat(1, 10))

    def AE_training(self,lr=0.000001):
        self.AE.train()
        for param in self.AE.parameters():
            param.requires_grad = True
        for param in self.FCM_Net.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(self.AE.parameters(), lr=lr, momentum=0.9)
        for x, target in self.data_loader:
            optimizer.zero_grad()
            x = Variable(x).cuda()
            y, u = self.AE(x.view(-1, self.feature_size))
            q = self.T_student(u.unsqueeze(1).repeat(1, 10, 1), self.u_mean.unsqueeze(0).repeat(self.batch_size, 1, 1))
            p = self.FCM_Net(u)
            loss = self.landa * nn.KLDivLoss()(q.log(), p) + nn.MSELoss()(x.view(-1, self.feature_size), y)
            loss.backward()
            optimizer.step()
        return self.AE


    def initialization(self):
        self.AE = torch.load('AE_MNIST_pretrain')
        self.AE.cuda()
        datas = torch.zeros([self.dataset_size, 10]).cuda()
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x.view(-1, self.feature_size))
            datas[(ii) * self.batch_size:(ii + 1) * self.batch_size] = u
            true_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1
        datas = datas.cpu()
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas.detach().numpy())
        self.u_mean = kmeans.cluster_centers_
        self.u_mean = torch.from_numpy(self.u_mean)
        self.u_mean = Variable(self.u_mean).cuda()

        return self.AE, self.u_mean

    def model_evaluation(self):
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            _, u = self.AE(x.view(-1, self.feature_size))
            p = self.FCM_Net(u)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()

            pred_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = y
            true_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1

        print('ACC', acc(true_labels,pred_labels,self.num_cluster))

    def train_IDECF(self):
      self.AE, self.u_mean = self.initialization()
      for T in range(0,self.max_iteration):
         print('Iteration: ', T+1)
         if T>0:
           self.umean = self.update_cluster_centers()

         self.FCM_Net = self.FCM_Net_train()
         self.AE = self.AE_training()
         self.model_evaluation()







