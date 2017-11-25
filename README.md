# Udacity project linear_algebra
## 机器学习入门-线性代数基础

---
运行测试代码：
`%run -i -e test.py LinearRegressionTestCase.test_...`
Ctrl + Enter 运行即可。

如果实现有问题，会有断言错误`AssertionError`被抛出。


以下是一些带有特定反馈的断言错误说明：
- AssertionError: Matrix A shouldn't be modified
  + 在实现augmentMatrix时修改了矩阵A
- AssertionError: Matrix A is singular
  + gj_Solve实现在矩阵A是奇异矩阵时没有返回None
- AssertionError: Matrix A is not singular
  + gj_Solve实现会在矩阵A不是奇异矩阵时返回None
- AssertionError: x have to be two-dimensional Python List
  + gj_Solve返回的数据结构不正确，x必须是二维列表，而且是Nx1的列向量
- AssertionError: Regression result isn't good enough
  + gj_Solve返回了计算结果，但是偏差过大
