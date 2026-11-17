# 运行与复核

要求 Python 3.11，仅标准库。项目使用命名空间包，从项目根目录运行，
不需要 setup 或安装。数据库、导出及测试副本放在调用方指定位置或测试临时目录。

完整回归：

    python3.11 -B -m unittest discover -s tests

核心训练、失败发布和固定交付：

    python3.11 -B -m unittest tests.test_daily_fit.TestDailyFit.test_numeric_fit_handcheck tests.test_fitting.TestFitting.test_constant_feature_and_solver_failure tests.test_publication.TestPublication.test_rollback_and_retry tests.test_delivery.TestDelivery.test_reopen_predict_export

修订原子性和历史重放：

    python3.11 -B -m unittest tests.test_measurements

固定队列和拒绝边界：

    python3.11 -B -m unittest tests.test_cohorts

设备隔离和行加权误差：

    python3.11 -B -m unittest tests.test_holdout

演示：

    python3.11 -B -m delivery.workflows

不带输出参数时，演示在独立临时目录实际创建 SQLite、导出 JSON、关闭重开，
比较预测后清理。输出 JSON 包含固定身份、训练诊断、留出误差、预测和复核清单。
指定 --database 与 --export 必须成对；--sample 可指定同格式项目输入。
重复运行相同样例幂等；源更正后运行旧样例只重放历史版本，不撤销更正。
每个 API 失败明确抛出异常，命令失败保留非零退出码。

测试对应：
- test_daily_fit：独立正交手算、未训练设备扰动、空训练及参数拒绝。
- test_measurements：批次冲突回滚、精确有理重放、倒序拒绝、撤回复制。
- test_cohorts：明确旧修订、更正撤回不漂移、成员与列顺序和拒绝边界。
- test_fitting：常量列、截距不惩罚、真实主元求解、奇异与溢出拒绝、
  公开求解失败端口不发布。
- test_holdout：部分设备交集拒绝、全部重复测量和行加权 SSE/MSE。
- test_publication：三个发布检查点回滚、磁盘关闭重开、重试和成功重放、
  重复初始化及不支持版本不清空、队列和拟合发布失败。
- test_delivery：真实每日流程、更正撤回和嵌套返回隔离、特征顺序、
  重开导出、非法预测、独立损坏副本父链拒绝及输出失败清理。

v1、v6 回归在真实断言之后输出 POINTWISE_SCENARIO 单行 JSON，
其中实际值来自业务返回，预期来自公开手算，不以标记替代断言或退出码。
其余测试通过正常 unittest 断言核验，无外部服务和模拟成功响应。

成果复核时先 read_package，再 export_package；发现完整性错误时保留原库供检查，
不要以当前源数据补齐旧成果。测试损坏只作用于测试临时目录中的独立副本。

## 相关噪声链路的核验与交接

独立核账（全部在 tests/test_correlated.py，不使用生产求解器作 oracle）：

    python3.11 -B -m unittest tests.test_correlated

该文件内自带主子式 PSD 判定、独立 C⁺ 构造（B(BᵀCB)⁻¹Bᵀ）和只靠 RREF
的可行域二次最小化 oracle；另用独立 RREF 复算条件均值与协方差。

相关样例核验与演示：

    python3.11 -B correlated_examples.py verify
    python3.11 -B correlated_examples.py demo

verify 先跑全部原回归再跑相关手算夹具；demo 在临时目录实际接纳、拟合、
条件预测、更正、导出并关闭重开，输出真实均值、完整协方差与复开一致性。
运行清单 .pointwise-runtime.json 的 verify/demo 已接这两个入口。

交接不变量：
- 训练噪声轴：fit.noise.revision_ids 恒等于 fit.scope.revision_ids（训练顺序）；
  相同轴重排请求规范到同一 fit_id，重复请求不新增成果。
- 支持约束：fit.residuals 精确满足 N·r=0（N 为 C 的零空间基），读取时复算；
  训练不可行整次 ValueError，归档不变。
- 预测消费：predict_correlated 复用同一固定拟合的 C、r 与训练顺序，不重拟合；
  完整联合协方差非半正定（哪怕形式 Schur 补为正）整次拒绝。
- 冻结性：后续更正/撤回、调用方改输入或返回对象、关闭重开均不改变旧拟合，
  相同请求返回相同预测；读取与导出不重拟合。
- 发布：相关拟合沿用原三检查点与 kind="fit"，异常原样传播并整体回滚。

复核相关成果时先 read_fit 核对父子链与残差支持，再 export_correlated_fit；
发现 ArchiveIntegrityError 时保留原库，不以当前源补齐。
