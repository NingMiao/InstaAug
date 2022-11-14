import torch


def get_data(model, loader, output_size, device, Li_configs={}, Li=None, mode='train'):
    """ encodes whole dataset into embeddings """
    
    n_copies=1
    aug=False
    average_feature=False
    if mode =='train':
        if Li_configs['li_flag'] and Li_configs['contrastive_train_aug']:
            aug=True
            n_copies=Li_configs['contrastive_train_copies']
            output_max=Li_configs['contrastive_train_output_max']
            if aug:
                average_feature=Li_configs['contrastive_train_feature_average']
    elif mode =='test':
        if Li_configs['li_flag'] and Li_configs['contrastive_test_aug']:
            aug=True
            n_copies=Li_configs['contrastive_test_copies']
            output_max=Li_configs['contrastive_test_output_max']
            if aug:
                average_feature=Li_configs['contrastive_test_feature_average']
        
    if average_feature:
        out_len=len(loader)
    else:
        out_len=len(loader)*n_copies
    
    xs = torch.empty(
        out_len, loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(out_len, loader.batch_size, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            if aug:
                x, logprob,_,_=Li(x, n_copies=n_copies, output_max=output_max)
            
            if average_feature:
                feature_list=[]
                for j in range(n_copies):
                    feature_list.append(model(x[loader.batch_size*j: loader.batch_size*(j+1)]).to(device))
                xs=torch.mean(torch.stack(feature_list, dim=0), dim=0)
                ys=y.to(device)
                
            else:
                for j in range(n_copies):
                    xs[i*n_copies+j] = model(x[loader.batch_size*j: loader.batch_size*(j+1)]).to(device)
                    ys[i*n_copies+j] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys
