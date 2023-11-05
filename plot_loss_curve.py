import seaborn as sns
import matplotlib.pyplot as plt

def smooth_loss(loss_values, alpha = 0.6):  # alpha平滑系数，取值范围为0到1之间
    smoothed_values = [loss_values[0]]

    for i in range(1, len(loss_values)):
        smoothed_value = alpha * smoothed_values[i-1] + (1 - alpha) * loss_values[i]
        smoothed_values.append(smoothed_value)

    return smoothed_values

def plt_curve(path):
    try:
        f = open(path)
    except FileNotFoundError:
        print('File not found!')
        raise

    lines = f.readlines()

    train_loss = []
    for line in lines:
        key_value = line.split(',')[2]
        if 'train_loss' in key_value:
            loss = float(key_value.split(' ')[-1])
            train_loss.append(loss)

    smoothed_loss = smooth_loss(train_loss)

    sns.set_theme(style="darkgrid")  # 设置主题为深色网格线
    sns.lineplot(x=range(len(train_loss)), y=train_loss, label='train_loss')
    sns.lineplot(x=range(len(smoothed_loss)), y=smoothed_loss, label='smoothed_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.title('Train loss curve')
    plt.show()


if __name__ == '__main__':
    plt_curve('./exps/r50_detr/log.txt')
