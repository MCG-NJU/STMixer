import datetime
import logging
import time

import torch

from alphaction.utils.metric_logger import MetricLogger
from alphaction.engine.inference import inference
from alphaction.utils.comm import synchronize, reduce_dict, all_gather
from alphaction.structures.memory_pool import MemoryPool
import torch.nn as nn

def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tblogger,
        start_val,
        val_period,
        dataset_names_val,
        data_loaders_val,
        distributed,
        mem_active,
        frozen_backbone_bn,
        output_folder,
):
    logger = logging.getLogger("alphaction.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    person_pool = arguments["person_pool"]
    model.train()
    if frozen_backbone_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
    start_training_time = time.time()
    end = time.time()
    losses_reduced = torch.tensor(0.0)

    for iteration, (slow_video, fast_video, whwh, boxes, labels, metadata, idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        slow_video = slow_video.to(device)
        if fast_video is not None:
            fast_video = fast_video.to(device)
        whwh = whwh.to(device)

        mem_extras = {}
        if mem_active:
            movie_ids = [m[0] for m in metadata]
            timestamps = [m[1] for m in metadata]
            mem_extras["person_pool"] = person_pool
            mem_extras["movie_ids"] = movie_ids
            mem_extras["timestamps"] = timestamps
            mem_extras["cur_loss"] = losses_reduced.item()


        loss_dict  = model(slow_video, fast_video, whwh, boxes, labels)
        losses = sum(loss_dict.values()) / len(loss_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict["total_loss"] = losses.detach().clone()
        loss_dict_reduced = reduce_dict(loss_dict)

        meters.update(**loss_dict_reduced)
        losses_reduced = loss_dict_reduced.pop("total_loss")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update mem pool
        if mem_active:
            pass

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if tblogger is not None:
                for name, meter in meters.meters.items():
                    tblogger.add_scalar(name, meter.median, iteration)
                tblogger.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)

        scheduler.step()

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration == max_iter:
            arguments.pop("person_pool", None)
            checkpointer.save("model_final", **arguments)

        if dataset_names_val and iteration > start_val and iteration % val_period == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # do validation
            val_in_train(
                model,
                dataset_names_val,
                data_loaders_val,
                tblogger,
                iteration,
                distributed,
                mem_active,
                output_folder
            )
            model.train()
            if frozen_backbone_bn:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
            torch.cuda.empty_cache()
            end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def val_in_train(
        model,
        dataset_names_val,
        data_loaders_val,
        tblogger,
        iteration,
        distributed,
        mem_active,
        output_folder,
):
    if distributed:
        model_val = model.module
    else:
        model_val = model
    for dataset_name, data_loader_val in zip(dataset_names_val, data_loaders_val):
        eval_res = inference(
            model_val,
            data_loader_val,
            dataset_name,
            mem_active,
            output_folder=output_folder,
        )
        synchronize()
        if tblogger is not None:
            eval_res, _ = eval_res
            total_mAP = eval_res['PascalBoxes_Precision/mAP@0.5IOU']
            tblogger.add_scalar(dataset_name + '_mAP_0.5IOU', total_mAP, iteration)