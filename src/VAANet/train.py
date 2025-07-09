from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

import time



def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, class_names, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()   # 新增：反向传播时间
    losses = AverageMeter()
    accuracies = AverageMeter()
    epoch_start_time = time.time()  # 新增：记录epoch开始时间
    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        start_time = time.time()
        output, loss = run_model(opt, [visual, target, audio], model, criterion, i, print_attention=False)
        forward_time.update(time.time() - start_time)
        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        start_time = time.time()
        # Backward and optimize
        optimizer.zero_grad()
        optimizer.backward(loss) 
        optimizer.step()
        backward_time.update(time.time() - start_time)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('train/batch/loss', losses.val, iter)
        writer.add_scalar('train/batch/acc', accuracies.val, iter)

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))

    # ---------------------------------------------------------------------- #
    epoch_time = time.time() - epoch_start_time  # 新增：纯epoch时间（含所有batch训练）
    pure_epoch_time = epoch_time - data_time.sum  # 新增：排除数据加载的纯训练时间
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Epoch Time (Pure Training): {:.2f}min".format(pure_epoch_time / 60))  
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))
    print("forward_time: {:.2f}min".format(forward_time.avg))
    print("Avg Backward Time per Batch: {:.4f}s".format(backward_time.avg))  

    writer.add_scalar('train/epoch/loss', losses.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies.avg, epoch)
