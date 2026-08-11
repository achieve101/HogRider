# 反馈式提交目录

本目录是可通过公开接口检查的零输出模板。请修改 `submission.py`，并把
自己的网络代码、配置和权重一起放入本目录。

最终打包前，还要在 `report/` 中放入且只放入一份正式 PDF 技术报告。

反馈接口固定为：

```python
process_sample(reference_sample, previous_error_sample)
```

其中 `previous_error_sample` 是 `e[t-1]`，不是当前 `e[t]`，也不是
未降噪期望噪声。

使用工具包打包后，文件会位于
`TeamID/Task2/submission.py`。
