from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy,batch_augment
import os
import time
import torch
from tqdm import tqdm
import jittor as jt
all_predictions = []  # 用于存储所有预测结果
output_file_path = "/home/ubuntu/zzq/CTEN_jittor/outcome3.txt"  # 保存预测结果的文件路径
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
    accuracies1 = AverageMeter()
    accuracies2 = AverageMeter()
    end_time = time.time()

    for i, data_item in tqdm(enumerate(data_loader)):
        visual, target, audio, visualization_item, batch_size,video_item = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with jt.no_grad():
            output1, loss1, gamma1 = run_model(opt, [visual, target, audio], model, criterion, i, print_attention=False)
            predicted_class =  jt.argmax(output1.data, 1)[0] 
            #print(output1)
            #print(predicted_class)
            all_predictions.append(predicted_class)
    #save_predictions_to_file(all_predictions, output_file_path)
    #print(f"预测结果已保存到 {output_file_path}")
            gamma_row_max = jt.max(gamma1,dim=1)[0]*0.7 + jt.min(gamma1,dim=1)[0]*0.3
            gamma_row_max = gamma_row_max.unsqueeze(0).transpose(1, 0)
            gamma_thre = gamma_row_max.expand(gamma1.shape)
            high_index = gamma1 < gamma_thre
            # output2,loss2,gamma2=output1,loss1,gamma1
            visual_erase1 = visual
            visual_erase1 = batch_augment(video_item, high_index, opt, visual_erase1)
            output2, loss2, gamma2 = run_model(opt, [visual_erase1, target, audio], model, criterion, i,print_attention=False)
        output=(output1+output2)/2.
        loss=loss1/2.+loss2/2.
        acc = calculate_accuracy(output, target)
        acc1=calculate_accuracy(output1,target)
        acc2=calculate_accuracy(output2,target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        accuracies1.update(acc1, batch_size)
        accuracies2.update(acc2, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
    Acc = max(accuracies.avg,accuracies1.avg,accuracies2.avg)
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    print("Val acc1: {:.4f}".format(accuracies1.avg))
    print("Val acc2: {:.4f}".format(accuracies2.avg))
    save_file_path = os.path.join(opt.ckpt_path, 'save_{}_{:.4f}.pth'.format(epoch,accuracies.avg))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict(),
    }
    #torch.save(states, save_file_path)
    return Acc