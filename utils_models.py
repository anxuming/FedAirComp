from utils_libs import *
import torchvision.models as models


class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)

        if self.name == 'mnist_air':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(100, 100)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(100, self.n_cls)
            self.tanh = nn.Tanh()
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)

        if self.name == 'emnist_CNN':
            self.n_cls = 47
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
            self.maxpool = nn.MaxPool2d(kernel_size=2)
            self.dropout1 = nn.Dropout(p=0.25)
            self.fc1 = nn.Linear(9216, 128)
            self.dropout2 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(128, self.n_cls)

        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'Cifar100_Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 100)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'tinyimagenet_Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 200)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'Densenet121':
            self.model = models.resnet18()
            self.model.fc = nn.Linear(512, 10)

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80

            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)

        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'mnist_air':
            x = x.view(-1, 1 * 28 * 28)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.tanh(x)

        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'emnist_CNN':
            x = self.conv1(x)
            x = self.conv2(x)
            # x = self.relu(x)
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.dropout1(x)
            x = torch.flatten(x, 1, -1)
            x = self.fc1(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)

        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'Resnet18' or self.name == 'Cifar100_Resnet18' or self.name == 'tinyimagenet_Resnet18':
            x = self.model(x)

        if self.name == 'Densenet121':
            x = self.model(x)

        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1, :, :]
            x = self.fc(last_hidden)

        return x
