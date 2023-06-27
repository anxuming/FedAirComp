from utils_general import *
from AirFast import *


def MGE(data_obj, model_func, init_model, local_learning_rate, com_amount, test_per,
        delta, ST, pm, lr_decay_per_round=1):
    n_client = data_obj.n_client
    client_x = data_obj.client_x
    client_y = data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)
    n_par = len(get_mdl_params([model_func()])[0])
    train_perf_sel = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    seed = 0
    Pm = pm
    Beta_max = np.ones(n_client, dtype=float)
    B_max = np.ones(n_client, dtype=float) * math.sqrt(Pm)
    for client in range(n_client):
        n_train = client_x[0].shape[0]
        Beta_max[client] = n_train / ST
    for i in range(com_amount):
        np.random.seed(seed + int(np.ceil(i / 20)))

        v = math.sqrt(2 / (4 - math.pi))
        Hn = np.random.rayleigh(scale=v, size=n_client)

        S = Beta_max / (Hn * B_max)
        H = torch.from_numpy(Hn).to(device)
        std = math.sqrt(delta)
        noise = torch.normal(0, std, size=(n_par,)).to(device)
        lr = local_learning_rate * (lr_decay_per_round ** i)
        a = min_Ea(Hn, B_max, Beta_max, S, delta)[1]
        E, Beta, Bn = Ea(Hn, B_max, Beta_max, delta, a)
        B = torch.from_numpy(Bn).to(device)
        M = torch.reshape(torch.mul(B, H), (1, -1)).to(device)
        for params in avg_model.parameters():
            params.requires_grad = True
        avg_model.train()
        grad = torch.zeros((n_client, n_par)).to(device)
        for client in range(n_client):
            train_x = client_x[client]
            train_y = client_y[client]
            n_train = client_x[0].shape[0]
            batch = int(round(n_train * Beta[client]))
            if batch == 0:
                continue
            train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=data_obj.dataset),
                                        batch_size=batch, shuffle=True, num_workers=1)
            train_gen_iter = train_gen.__iter__()
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_ref = avg_model(batch_x)
            loss = loss_fn(y_ref, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=avg_model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in avg_model.named_parameters():
                temp = param.grad.reshape(-1)
                grad[client][idx: idx + len(temp)] = temp
                idx += len(temp)
            avg_model.zero_grad()
        grad = grad.double()
        with torch.no_grad():
            grad_avg = a * lr * (torch.mm(M, grad.reshape(n_client, n_par)) + noise).squeeze(0)
            idx = 0
            for name, param in avg_model.named_parameters():
                length = param.reshape(-1).shape[0]
                param -= grad_avg[idx: idx + length].reshape(param.shape)
                idx += length
        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.test_x, data_obj.test_y, avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f."
                  % (i + 1, acc_test, loss_test), flush=True)

            loss_test, acc_test = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f." % (i + 1, acc_test, loss_test),
                  flush=True)

        for params in avg_model.parameters():
            params.requires_grad = False

    return


def AirCOP(data_obj, learning_rate, com_amount, test_per, delta, model_func,
           init_model, pm, lr_decay_per_round=1):
    n_client = data_obj.n_client

    client_x = data_obj.client_x
    client_y = data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)

    train_perf_sel = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    seed = 0
    Pm = pm
    D = np.zeros(n_client, dtype=float)
    for client in range(n_client):
        D[client] = client_x[client].shape[0]
    for i in range(com_amount):
        np.random.seed(seed + int(np.ceil(i / 20)))
        v = math.sqrt(2 / (4 - math.pi))
        Hn = np.random.rayleigh(scale=v, size=n_client)
        H = torch.from_numpy(Hn).to(device)
        std = math.sqrt(delta)
        noise = torch.normal(0, std, size=(n_par,)).to(device)
        Y, a, Bn = cop(math.sqrt(Pm), Hn, D, delta)
        B = torch.from_numpy(Bn).to(device)
        M = torch.reshape(torch.mul(B, H), (1, -1)).to(device)
        for params in avg_model.parameters():
            params.requires_grad = True
        avg_model.train()
        grad = torch.zeros((n_client, n_par)).to(device)
        lr = learning_rate * (lr_decay_per_round ** i)
        for client in range(n_client):
            train_x = client_x[client]
            train_y = client_y[client]
            n_train = client_x[0].shape[0]
            train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=data_obj.dataset),
                                        batch_size=n_train, shuffle=True, num_workers=1)
            train_gen_iter = train_gen.__iter__()
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_ref = avg_model(batch_x)
            loss = loss_fn(y_ref, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=avg_model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in avg_model.named_parameters():
                temp = param.grad.reshape(-1)
                grad[client][idx: idx + len(temp)] = temp
                idx += len(temp)
            avg_model.zero_grad()
        grad = grad.double()
        with torch.no_grad():
            grad_avg = a * lr * (torch.mm(M, grad.reshape(n_client, n_par)) + noise).squeeze(0)
            idx = 0
            for name, param in avg_model.named_parameters():
                length = param.reshape(-1).shape[0]
                param -= grad_avg[idx: idx + length].reshape(param.shape)
                idx += length
        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.test_x, data_obj.test_y, avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f."
                  % (i + 1, acc_test, loss_test), flush=True)

            loss_test, acc_test = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f." % (i + 1, acc_test, loss_test),
                  flush=True)

        for params in avg_model.parameters():
            params.requires_grad = False

    return


def AirAvg(data_obj, learning_rate, com_amount, test_per, delta, model_func,
           init_model, pm, lr_decay_per_round=1):
    n_client = data_obj.n_client

    client_x = data_obj.client_x
    client_y = data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)

    train_perf_sel = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    seed = 0
    Pm = pm
    Beta = np.ones(n_client, dtype=float) * (1 / n_client)
    B_max = np.ones(n_client, dtype=float) * math.sqrt(Pm)
    for i in range(com_amount):
        np.random.seed(seed + int(np.ceil(i / 20)))

        v = math.sqrt(2 / (4 - math.pi))
        Hn = np.random.rayleigh(scale=v, size=n_client)

        H = torch.from_numpy(Hn).to(device)
        std = math.sqrt(delta)
        noise = torch.normal(0, std, size=(n_par,)).to(device)
        Bl = []
        for j in range(n_client):
            b = min(Beta[j] * n_client / Hn[j], B_max[j])
            Bl.append(b)
        Bn = np.array(Bl)
        B = torch.from_numpy(Bn).to(device)
        M = torch.reshape(torch.mul(B, H), (1, -1)).to(device)
        for params in avg_model.parameters():
            params.requires_grad = True
        avg_model.train()
        grad = torch.zeros((n_client, n_par)).to(device)
        lr = learning_rate * (lr_decay_per_round ** i)
        for client in range(n_client):
            train_x = client_x[client]
            train_y = client_y[client]
            n_train = client_x[0].shape[0]
            train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=data_obj.dataset),
                                        batch_size=n_train, shuffle=True, num_workers=1)
            train_gen_iter = train_gen.__iter__()
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_ref = avg_model(batch_x)
            loss = loss_fn(y_ref, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=avg_model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in avg_model.named_parameters():
                temp = param.grad.reshape(-1)
                grad[client][idx: idx + len(temp)] = temp
                idx += len(temp)
            avg_model.zero_grad()
        grad = grad.double()
        with torch.no_grad():
            grad_avg = 1 / n_client * lr * (torch.mm(M, grad.reshape(n_client, n_par)) + noise).squeeze(0)
            idx = 0
            for name, param in avg_model.named_parameters():
                length = param.reshape(-1).shape[0]
                param -= grad_avg[idx: idx + length].reshape(param.shape)
                idx += length
        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.test_x, data_obj.test_y, avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f."
                  % (i + 1, acc_test, loss_test), flush=True)

            loss_test, acc_test = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f." % (i + 1, acc_test, loss_test),
                  flush=True)

        for params in avg_model.parameters():
            params.requires_grad = False

    return


def AirTPC(data_obj, model_func, init_model, local_learning_rate, com_amount, test_per,
           delta, pm, lr_decay_per_round=1):
    n_client = data_obj.n_client

    client_x = data_obj.client_x
    client_y = data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)

    train_perf_sel = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    seed = 0
    Pm = pm
    for i in range(com_amount):
        np.random.seed(seed + int(np.ceil(i / 20)))
        v = math.sqrt(2 / (4 - math.pi))
        Hn = np.random.rayleigh(scale=v, size=n_client)
        H = torch.from_numpy(Hn).to(device)
        std = math.sqrt(delta)
        noise = torch.normal(0, std, size=(n_par,)).to(device)
        a, Bn, Y = TPC(math.sqrt(Pm), Hn, delta)
        a = 1 / (n_client * math.sqrt(a))
        B = torch.from_numpy(Bn).to(device)
        M = torch.reshape(torch.mul(B, H), (1, -1)).to(device)
        for params in avg_model.parameters():
            params.requires_grad = True
        avg_model.train()
        grad = torch.zeros((n_client, n_par)).to(device)
        lr = local_learning_rate * (lr_decay_per_round ** i)
        for client in range(n_client):
            train_x = client_x[client]
            train_y = client_y[client]
            n_train = client_x[0].shape[0]
            train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=data_obj.dataset),
                                        batch_size=n_train, shuffle=True, num_workers=1)
            train_gen_iter = train_gen.__iter__()
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_ref = avg_model(batch_x)
            loss = loss_fn(y_ref, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=avg_model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in avg_model.named_parameters():
                temp = param.grad.reshape(-1)
                grad[client][idx: idx + len(temp)] = temp
                idx += len(temp)
            avg_model.zero_grad()
        grad = grad.double()
        with torch.no_grad():
            grad_avg = a * lr * (torch.mm(M, grad.reshape(n_client, n_par)) + noise).squeeze(0)
            idx = 0
            for name, param in avg_model.named_parameters():
                length = param.reshape(-1).shape[0]
                param -= grad_avg[idx: idx + length].reshape(param.shape)
                idx += length
        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.test_x, data_obj.test_y, avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f."
                  % (i + 1, acc_test, loss_test), flush=True)

            loss_test, acc_test = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f." % (i + 1, acc_test, loss_test),
                  flush=True)

        for params in avg_model.parameters():
            params.requires_grad = False

    return


def FedSGD(data_obj, model_func, init_model, local_learning_rate, com_amount, test_per,
           lr_decay_per_round=1):
    n_client = data_obj.n_client
    client_x = data_obj.client_x
    client_y = data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)
    n_par = len(get_mdl_params([model_func()])[0])
    train_perf_sel = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    for i in range(com_amount):
        for params in avg_model.parameters():
            params.requires_grad = True
        avg_model.train()
        grad = torch.zeros((n_client, n_par)).to(device)
        lr = local_learning_rate * (lr_decay_per_round ** i)
        for client in range(n_client):
            train_x = client_x[client]
            train_y = client_y[client]
            n_train = client_x[0].shape[0]
            train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=data_obj.dataset),
                                        batch_size=n_train, shuffle=True, num_workers=1)
            train_gen_iter = train_gen.__iter__()
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_ref = avg_model(batch_x)
            loss = loss_fn(y_ref, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=avg_model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in avg_model.named_parameters():
                temp = param.grad.reshape(-1)
                grad[client][idx: idx + len(temp)] = temp
                idx += len(temp)
            avg_model.zero_grad()
        grad = grad.double()
        with torch.no_grad():
            grad_avg = lr * torch.mean(grad.reshape(n_client, n_par), dim=0, keepdim=True).squeeze(0)
            idx = 0
            for name, param in avg_model.named_parameters():
                length = param.reshape(-1).shape[0]
                param -= grad_avg[idx: idx + length].reshape(param.shape)
                idx += length
        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(data_obj.test_x, data_obj.test_y, avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f."
                  % (i + 1, acc_test, loss_test), flush=True)

            loss_test, acc_test = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f." % (i + 1, acc_test, loss_test),
                  flush=True)

        # Freeze model
        for params in avg_model.parameters():
            params.requires_grad = False

    return
