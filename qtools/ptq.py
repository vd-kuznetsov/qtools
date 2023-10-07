# evaluate taken from here https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
import random

import numpy as np
import pandas as pd
import torch
from torch.ao import quantization
from torch.ao.quantization import quantize_fx

from .metrics import AverageMeter, accuracy, inference_time, size_of_model


def ptqs_pipeline(
    baseline,
    eval_dataloader,
    train_dataloader=None,
    tag="fx",
    seed=42,
    backend="x86",
    qconfig=None,
    num_calibration_batches=10,
    num_batches=None,
    baseline_info=True,
    save_predict=False,
):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = baseline.to("cpu")
    model.eval()

    if train_dataloader is None:
        train_dataloader = eval_dataloader

    if num_batches is None:
        num_batches = len(eval_dataloader)

    sample = next(iter(train_dataloader))
    data_keys = list(sample.keys()) if isinstance(sample, dict) else [0, 1]

    batch_size = sample[data_keys[0]].shape[0]

    if baseline_info:
        text = "Before quantization"
        model_metrics(
            text, model, eval_dataloader, data_keys, num_batches, batch_size, save_predict
        )

    if qconfig is None:
        qconfig = quantization.get_default_qconfig(backend)

    print("\nStart Post Training Quantization")

    # Specify quantization configuration and prepare
    torch.backends.quantized.engine = backend
    if tag == "fx":
        qconfig_dict = {"": qconfig}
        example_inputs = (sample[data_keys[0]][0:1],)
        model = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs)
    else:
        # Fuse Conv, bn, relu and etc
        model.fuse_model()
        model.qconfig = qconfig
        quantization.prepare(model, inplace=True)
    print("Post Training Quantization: Prepare done")

    # Calibrate with the representative set
    print("Post Training Quantization: Start calibration")
    evaluate(model, train_dataloader, data_keys, neval_batches=num_calibration_batches)
    print("\nPost Training Quantization: Calibration done")

    if tag == "fx":
        model = quantize_fx.convert_fx(model)
    else:
        quantization.convert(model, inplace=True)
    print("Post Training Quantization: Convert done")

    text = "After quantization"
    model_metrics(
        text, model, eval_dataloader, data_keys, num_batches, batch_size, save_predict
    )

    return model


def evaluate(model, dataloader, data_keys, neval_batches):
    model.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    outputs_array = np.array([])
    image_key = data_keys[0]
    label_key = data_keys[1]
    cnt = 0
    with torch.no_grad():
        for batch in dataloader:
            image, target = batch[image_key], batch[label_key]
            output = model(image)
            output_batch = torch.max(output, 1)[1]
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print(".", end="")
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            outputs_array = np.hstack((outputs_array, output_batch))
            if cnt >= neval_batches:
                return top1, top5, outputs_array

    return top1, top5, outputs_array


def model_metrics(
    text, model, dataloader, data_keys, num_batches, batch_size, save_predict=False
):
    print(text)
    size_of_model(model)
    inference_time(model, dataloader, data_keys)
    print("Metrics are being calculated..")

    top1, top5, outputs_array = evaluate(
        model, dataloader, data_keys, neval_batches=num_batches
    )
    print(
        "\nEvaluation accuracy on %d images, top1 = %2.2f"
        % (num_batches * batch_size, top1.avg)
    )
    print(
        "Evaluation accuracy on %d images, top5 = %2.2f"
        % (num_batches * batch_size, top5.avg)
    )

    if save_predict:
        outputs_df = pd.DataFrame(outputs_array, columns=["predict"])
        text = text.lower().replace(" ", "_")
        outputs_df.to_csv("model_predict_" + text + ".csv", index=False)
