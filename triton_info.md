# Triton information

Metrics values obtained before deadline, [proof](./images/proof.png).

Run `dvc pull` to download the onnx model to the correct location.

## System configuration

| OS           | CPU                                     | vCPU | RAM    |
| ------------ | --------------------------------------- | ---- | ------ |
| Ubuntu 22.04.3 | Intel Xeon Gold 6136 v4 | 24    | 256 GiB |

In [docker-compose.yml](./triton/docker-compose.yml) there is a limit of up to 10 cores on the CPU.

## Task description

Classification of 10 classes at [ImageWoof](https://github.com/fastai/imagenette#imagewoof)

## Model repository structure

```bash
triton/model_repository
└── onnx-resnet18
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## Metrics

```bash
perf_analyzer -m onnx-resnet18 -u localhost:8500 --concurrency-range 1:5 --shape IMAGES:1,3,224,224
```

Baseline: use instance_groups count = 1 and empty dynamic_batching

Performed experiments:

| Optimization       | Conq Range | **Throughput, fps** | **Latency, ms** |
|--------------------|------------|---------------------|-----------------|
| Baseline           | $1$        | $59.27$             | $16.9$          |
| Baseline           | $2$        | $77.54$             | $25.7$          |
| Baseline           | $3$        | $76.32$             | $39.3$          |
| Baseline           | $4$        | $75.93$             | $52.6$          |
| Baseline           | $5$        | $77.59$             | $64.5$          |
|                    |            |                     |                 |
| More instances (2) | $1$        | $23.77$             | $42.1$          |
| More instances (2) | $2$        | $35.44$             | $56.3$          |
| More instances (2) | $3$        | $40.65$             | $74.0$          |
| More instances (2) | $4$        | $39.11$             | $102.0$         |
| More instances (2) | $5$        | $39.61$             | $126.2$         |
|                    |            |                     |                 |
| More instances (3) | $1$        | $9.72$              | $102.9$         |
| More instances (3) | $2$        | $15.59$             | $129.0$         |
| More instances (3) | $3$        | $22.89$             | $131.0$         |
| More instances (3) | $4$        | $23.11$             | $173.3$         |
| More instances (3) | $5$        | $25.27$             | $196.9$         |

The optimal is Baseline, let's try to parameterize the max_queue_delay_microseconds parameter of dynamic_batching:

| max_queue_delay_microseconds | Conq Range | **Throughput, fps** | **Latency, ms** |
|--------------------|------------|---------------------|-----------------|
| 500           | $1$        | $63.97$             | $15.63$          |
| 500           | $2$        | $74.65$             | $26.78$          |
| 500           | $3$        | $77.38$             | $38.76$          |
| 500           | $4$        | $76.43$             | $52.33$          |
| 500           | $5$        | $75.03$             | $66.62$          |
|                    |            |                     |                 |
| 1000          | $1$        | $59.92$             | $16.69$           |
| 1000          | $2$        | $78.93$             | $25.29$           |
| 1000          | $3$        | $81.97$             | $36.63$           |
| 1000          | $4$        | $82.54$             | $48.45$           |
| 1000          | $5$        | $79.70$             | $62.07$           |
|                    |            |                     |                 |
| 2000          | $1$        | $60.99$             | $16.39$           |
| 2000          | $2$        | $81.15$             | $24.63$           |
| 2000          | $3$        | $78.82$             | $38.05$           |
| 2000          | $4$        | $79.15$             | $50.52$           |
| 2000          | $5$        | $77.88$             | $64.19$           |

## Conclusion

Increasing the number of instances gives a decrease in throughput and an increase in latency relative to the base configuration.

Completed configuration of the enumerated parameters for [config.pbtxt](./triton/model_repository/onnx-resnet18/config.pbtxt):

```
instance_group [
    {
        count: 1
    }
]

dynamic_batching {
    max_queue_delay_microseconds: 1000
}
```
