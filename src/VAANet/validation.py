from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

import os
import time
from tqdm import tqdm
import jittor as jt

all_predictions = []  # 用于存储所有预测结果
output_file_path = "/home/ubuntu/wwc/zzq/VAANet_jt/outcome5.txt"  # 保存预测结果的文件路径
def save_predictions_to_file(predictions, file_path):
    """
    将预测结果保存到文件中。
    
    :param predictions: 预测结果列表，每个元素是一个预测类别的数组。
    :param file_path: 保存文件的路径。
    """
    with open(file_path, 'w') as f:
        for pred in predictions:
            # 将预测结果转换为字符串并写入文件
            f.write(f"{pred.tolist()}\n")


def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with jt.no_grad():
            output, loss = run_model(opt, [visual, target, audio], model, criterion, i)
            print(output)
            predicted_class =  jt.argmax(output.data, 1)[0] 
            all_predictions.append(predicted_class)
    #save_predictions_to_file(all_predictions, output_file_path)
    #print(f"预测结果已保存到 {output_file_path}")

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))

    save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    #torch.save(states, save_file_path)
