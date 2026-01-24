# TODO:

- (tmp) we encourage contributions from the OSS community, but due to having limited maintainers, all PRs must be tied to an existing issue. if no issue exists yet, open one up & discuss ur changes before submitting a PR. all PRs that don’t follow these rules will be rejected automatically without human review
- describe project structure
  - removed module level READMEs. document module details here

## known issues:

```
2026-01-24 03:52:38.996928: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
2026-01-24 03:52:38.997040: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: CANCELLED: GetNextFromShard was cancelled
         [[{{node MultiDeviceIteratorGetNextFromShard}}]]
         [[RemoteCall]] [type.googleapis.com/tensorflow.DerivedStatus='']
...
```

insert snippet about TF prefetch threads
