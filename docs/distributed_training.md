### distributed training

#### pseudo distributed training in single node

- single script

```shell
./scripts/train.sh test_symbol local
```

- distributed training by horovod

```shell
./scripts/horovodrun.sh 8 ./hosts/local ./scripts/train.sh test_symbol
```

- Distributed training by kv_store

```shell
./scripts/launch.sh 1 hosts/local ./scripts/train.sh test_symbol
```

#### distributed training

assume your have 2 nodes, each node has 8 gpus, and the node info in `hosts/dist2` as below:

```shell
192.168.0.1
192.168.0.2
```

- Distributed training by horovod

```shell
./scripts/horovodrun.sh 16 ./hosts/dist2 ./scripts/train.sh test_symbol
```

- distributed training by kv_store

```shell
./scripts/launch.sh 2 hosts/local ./scripts/train.sh test_symbol
```