import torch
import torch.nn as nn
import torch.optim as optim


def eval_sgd(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size, topk=[1, 5], epoch=5000, Li_configs={}, Li=None, eval_start=500, eval_every=100):
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    
    acc_top1_old=0
    flag=0
    for ep in range(epoch):
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()
        
        if ep>=eval_start and ep%eval_every==0:
            clf.eval()
            with torch.no_grad():
                y_pred = clf(x_test)
            
            if Li_configs['li_flag'] and Li_configs['contrastive_test_aug'] and not Li_configs['contrastive_test_feature_average']:
                y_pred=y_pred.reshape([-1, Li_configs['contrastive_test_copies'], test_batch_size, num_class])
                y_pred=torch.sum(torch.exp(y_pred), dim=1).reshape([-1, num_class]) #!Not using logprob information yet
                y_test=y_test.reshape([-1, Li_configs['contrastive_test_copies'], test_batch_size])[:,0,:].reshape([-1])
    
    
            pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
            acc = {
                t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
                for t in topk
            }
            
            if acc[1]>acc_top1_old:
                acc_top1_old=acc[1]
                flag=0
                acc_best=acc
            else:
                flag+=1
                if flag>=2:
                    break
            clf.train()
            
            print(acc, ep)
    
    
    del clf
    return acc_best
