from utils_general import *
from utils_methods import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'],
                    type=str, default='CIFAR10')
parser.add_argument('--non-iid', action='store_true', default=False)
parser.add_argument('--rule', choices=['Drichlet', 'n_cls'], type=str, default='Drichlet')
parser.add_argument('--rule-arg', default=6.0, type=float)  # Drichlet 与分布相关的参数
parser.add_argument('--act_prob', default=1.0, type=float)  # 每个用户参与通信的概率
parser.add_argument('--method', choices=['MGE', 'AirCOP', 'AirTPC', 'AirAvg', 'FedSGD'], type=str, default='MGE')
parser.add_argument('--n_client', default=20, type=int)
parser.add_argument('--epochs', default=1200, type=int)  # 通信次数（大循环数）
parser.add_argument('--local-learning-rate', default=0.1, type=float)
parser.add_argument('--lr-decay', default=0.998, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pm', default=5.0, type=float)
parser.add_argument('--S_T', default=40000, type=int)
parser.add_argument('--test-per', default=1, type=int)
args = parser.parse_args()
print(args)

# Dataset initialization
data_path = './'

# 系统中用户的数量
n_client = args.n_client
# Generate IID or Dirichlet distribution
if args.non_iid is False:
    data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                             data_path=data_path)
elif args.rule == 'Drichlet':
    data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='Drichlet',
                             rule_arg=args.rule_arg, data_path=data_path)
elif args.rule == 'n_cls':
    data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='n_cls',
                             rule_arg=args.rule_arg, data_path=data_path)

if args.dataset == 'CIFAR10':
    model_name = 'Resnet18'
elif args.dataset == 'CIFAR100':
    model_name = 'Cifar100_Resnet18'

# Common hyperparameters
com_amount = args.epochs
save_period = 10000
weight_decay = 5e-4
# weight_decay = 0
act_prob = args.act_prob
suffix = model_name
lr_decay_per_round = args.lr_decay
pm = args.pm

# Model function
model_func = lambda: client_model(model_name)

# Initialize the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(0)
init_model = model_func()

if args.method == 'MGE':
    learning_rate = args.local_learning_rate
    test_per = args.test_per
    S_T = args.S_T

    MGE(data_obj=data_obj, model_func=model_func, init_model=init_model, local_learning_rate=learning_rate,
        com_amount=com_amount, test_per=test_per, delta=0.1, ST=S_T, pm=pm, lr_decay_per_round=lr_decay_per_round)

elif args.method == 'AirCOP':
    learning_rate = args.local_learning_rate
    test_per = args.test_per
    AirCOP(data_obj=data_obj, learning_rate=learning_rate, com_amount=com_amount, test_per=test_per, delta=0.1,
           model_func=model_func, init_model=init_model, pm=pm, lr_decay_per_round=lr_decay_per_round)


elif args.method == "AirTPC":
    learning_rate = args.local_learning_rate
    test_per = args.test_per
    AirTPC(data_obj=data_obj, model_func=model_func, init_model=init_model, local_learning_rate=learning_rate,
           com_amount=com_amount, test_per=test_per, delta=0.1, pm=pm, lr_decay_per_round=lr_decay_per_round)


elif args.method == 'AirAvg':
    learning_rate = args.local_learning_rate
    test_per = args.test_per
    AirAvg(data_obj=data_obj, model_func=model_func, init_model=init_model, learning_rate=learning_rate,
           com_amount=com_amount, test_per=test_per, delta=0.1, pm=pm, lr_decay_per_round=lr_decay_per_round)

elif args.method == 'FedSGD':
    epoch = args.local_epochs
    learning_rate = args.local_learning_rate
    test_per = args.test_per

    FedSGD(data_obj=data_obj, model_func=model_func, init_model=init_model, local_learning_rate=learning_rate,
           com_amount=com_amount, test_per=test_per, lr_decay_per_round=lr_decay_per_round)
