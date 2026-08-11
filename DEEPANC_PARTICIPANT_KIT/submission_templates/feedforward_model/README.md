# 前馈式提交目录

本目录是可通过公开接口检查的零输出模板。请修改 `submission.py`，并把
自己的网络代码、配置和权重一起放入本目录。

最终打包前，还要在 `report/` 中放入且只放入一份正式 PDF 技术报告。

创建到工具包根目录后，入口格式为：

```text
my_submission.submission:create_model
```

不要删除 `submission.py`、`create_model()`、`sample_rate`、
`requires_error`、`reset()`、`process_sample()` 或
`get_complexity()`。

使用工具包打包后，文件会位于
`TeamID/Task2/submission.py`。
