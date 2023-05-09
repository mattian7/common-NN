import argparse
from model import *
from utils import *
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import random
from torchvision.transforms import ToTensor
import torchvision


def parse_arguments():
    parser = argparse.ArgumentParser(description="parameter of mnist-classifiers")
    parser.add_argument("--model", type=str, default="LeNet", choices=['LeNet', 'ResNet', 'AlexNet', 'DenseNet', 'MobileNet'])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    train_dataset = MNISTDataset(root='data', train=True, transform=ToTensor())
    train_num = train_dataset.__len__()
    indices = [index for index in range(train_num)]
    random.shuffle(indices)
    valid_num = int(train_num/10)
    train_num = train_num - valid_num
    train_indices = indices[:train_num]
    valid_indices = indices[train_num:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)


    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batchsize)
    valid_dataloader = DataLoader(train_dataset, sampler=valid_sampler, batch_size=args.batchsize)

    test_dataset = MNISTDataset(root='data', train=False, transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

    output_dir = 'log/' + args.model + '.txt'
    # 判断是否是LeNetorMobileNet，如果是，则无需放大图像。
    flag = 0

    if args.model == 'LeNet':
        model = LeNet(args).cuda()
        flag = 1
    elif args.model == 'ResNet':
        model = ResNet(args).cuda()
    elif args.model == 'AlexNet':
        model = AlexNet(args).cuda()
    elif args.model == 'DenseNet':
        model = DenseNet(growth_rate=16, input_features=64, num_layers=[6, 12, 24, 16], num_classes=10).cuda()
    elif args.model == 'MobileNet':
        model = MobileNet(args).cuda()
        flag = 1
    else:
        raise ValueError("model is not properly defined ...")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 定义优化器
    with open(output_dir, "w") as wp:
        write_steup(args, wp)
        for epoch in range(args.num_epochs):
            valid_num = 0
            valid_correct = 0
            wp.write("epoch "+str(epoch)+":\n")
            for iter, data in enumerate(train_dataloader):
                X_train, Y_train = data
                # 把28x28的图像放大到AlexNet的输入大小
                if flag != 1:
                    X_train = torchvision.transforms.Resize(227)(X_train)
                x_train, y_train = X_train.cuda(), Y_train.cuda()
                optimizer.zero_grad()

                output = model(x_train)
                loss = loss_function(output, y_train)
                loss.backward()
                optimizer.step()

                if iter % 100 == 0:
                    print("epoch\t{}\t{}/938\tLoss: {}".format(epoch, iter, loss.item()))
                    wp.write(str(iter)+'/938\tLoss: '+str(loss.item())+'\n')

            with torch.no_grad():
                for iter, data in enumerate(valid_dataloader):
                    X_valid, Y_valid = data
                    if flag != 1:
                        X_valid = torchvision.transforms.Resize(227)(X_valid)
                    x_valid, y_valid = X_valid.cuda(), Y_valid.cuda()
                    output = model(x_valid)
                    pred = output.max(1)[1]
                    valid_correct += (pred == y_valid).sum().item()
                    valid_num += x_valid.shape[0]
            acc = (valid_correct * 1.0 / valid_num)*100

            print("Valid @ epoch\t{}\tAcc: {}%".format(epoch, acc))
            wp.write("Valid @ epoch\t" + str(epoch) + '\tAcc: ' + str(acc)+'\n')

        print("\n....Testing....\n")
        test_correct = 0
        test_num = 0
        with torch.no_grad():
            for iter, data in enumerate(test_dataloader):
                X_test, Y_test = data
                if flag != 1:
                    X_test = torchvision.transforms.Resize(227)(X_test)
                x_test, y_test = X_test.cuda(), Y_test.cuda()
                output = model(x_test)
                pred = output.max(1)[1]
                test_correct += (pred == y_test).sum().item()
                test_num += x_test.shape[0]
        test_acc = (test_correct * 1.0 / test_num)*100

        print("Test Acc: {}%".format(test_acc))
        wp.write("Test acc: "+str(test_acc)+'\n')


if __name__ == "__main__":
    main()

