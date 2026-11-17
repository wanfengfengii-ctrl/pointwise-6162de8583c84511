# 相关测量拟合与条件联合预测

## 当前工程与本次交付

README.md、RUNBOOK.md 和现有运行清单描述完整 V1。保留所有旧接口、旧成果编码和全部原测试；不要改变 V1 的标准化、求解依赖替换或留出评分语义。本次新增的是已知测量噪声下的相关拟合，以及消费该固定拟合的条件联合预测，不估计噪声参数，不计算参数估计不确定性，也不新增留出评估或模型选择。

以下方法添加到 delivery.app.NumericArchiveApp。业务实现应落在现有拟合生产和预测执行职责中；底层表示、内部辅助函数及分解是否缓存由实现决定。允许全量重算和精确消元，不设性能指标。不得仅包装 fit_numeric。

## 数值与训练范围

新方法的数值参数只接受非布尔 int、Fraction 或能被 Fraction 解释的有理字符串，拒绝 float、非法字符串及其他类型。既有 apply_measurements 的接纳规则不变；新拟合直接使用固定修订中保存的精确有理载荷，不经过 float 转换。矩阵为列表的列表，行输入为列表的列表；维度、身份和数值非法均抛 ValueError，不发布成果。

fit_correlated(cohort_id, devices, lambda_, noise) 返回固定拟合字典。
devices 沿用 V1 设备选择规则，包含所选设备在该队列中的全部行，训练顺序为队列中这些行的顺序。训练限 1～24 行、1～4 个原始数值特征。lambda_ 为严格正有理数。
noise 必须恰含 revision_ids 和 covariance 两个键。revision_ids 是与全部训练修订一一对应的字符串列表，允许重排；遗漏、重复、外来修订均拒绝。covariance 的两轴依该列表排列，必须为 n×n 对称半正定矩阵。接纳后按训练顺序同步重排两轴和身份。对称性、半正定性、秩、支持和可行性均按精确有理数判断；不使用容差、截断特征值或对角扰动。

令 X 为按 feature_order 取原始特征并在首列加 1 的设计矩阵；y 为训练目标，C 为规范顺序的噪声协方差，r=y-Xβ。使用原始特征及截距，不沿用 V1 标准化；常量特征仍保留原值。
在 r∈im(C) 下唯一最小化
J = rᵀC⁺r/n + lambda_ × Σ(j≥1) β_j²。
C⁺ 表示 Moore–Penrose 逆，截距不惩罚。奇异矩阵和零矩阵均允许；零噪声方向是硬约束。数据不可行时抛 ValueError，归档不变；不能以最小二乘近似代替支持可行性。

返回字典至少包含：
- model_kind="correlated"、fit_id、cohort_id、scope、feature_order；scope 使用 V1 的设备和完整有序训练修订语义。
- lambda：Fraction；noise={revision_ids,covariance}，规范顺序与 scope.revision_ids 相同，协方差元素为 Fraction。
- coefficients：按 [截距, feature_order 各系数] 排列的 Fraction 列表；residuals：训练顺序的 Fraction 列表。
- diagnostics={n,noise_rank,loss,penalty,objective}；前两项为整数，后三项为 Fraction，分别为 rᵀC⁺r/n、正则项和二者之和。
允许增加内部所需字段，但不能改变上述含义。成功重复同一规范请求返回同一 fit_id，不新增重复成果；相同轴重排请求须规范到同一拟合身份。

## 条件联合预测

predict_correlated(fit_id, rows, B, D) 返回 {mean,covariance}。
fit_id 必须指向本题固定相关拟合，不能用 V1 拟合冒充。rows 是至多 8 条按原始 feature_order 排列的新特征行。m 为新行数，B 为 m×n 的新—训练交叉协方差，列按返回拟合的训练顺序排列；D 为 m×m 新测量噪声协方差，按 rows 顺序排列。
必须精确核验完整联合矩阵 [[C,Bᵀ],[B,D]] 对称半正定，不能只核验 D 或形式上的 Schur 补。将拟合 β 当作固定参数，返回
mean = X_new β + BC⁺r，
covariance = D - BC⁺Bᵀ。
均值列表和完整协方差矩阵的元素均为 Fraction，包含新行之间的交叉项；不加入系数估计不确定性，不得逐行条件化后丢弃交叉项。预测本身不创建新归档成果。m=0 时 rows=B=D=[]，仍核验 fit_id，返回 {mean:[],covariance:[]}。其他错误整次抛 ValueError，不返回部分结果。

V1 的 evaluate_holdout 和 predict_numeric 仍只处理原预测链；evaluate_holdout 收到相关 fit_id 时抛 ValueError，不要求生成相关留出包。

## 固定成果与交付

拟合固定完整有序训练修订、原始列、C、lambda、系数和残差；拟合生产与预测消费必须通过同一冻结训练噪声和残差语义交接。
扩展 read_fit 读取上述固定字典，V1 返回行为不变。后续源更正或撤回、调用方修改输入或返回嵌套对象、关闭重开，均不得改变旧拟合及相同请求的预测。读取和导出不重新拟合，不按当前源头补齐旧成果。
相关拟合沿用原 publication_hook(stage,artifact_id,kind)，kind 为 fit，三个原检查点都须生效；任一检查点异常须原样传播、整体回滚，旧成果不变，移除故障后可重试。成功重放仍幂等。
新增 export_correlated_fit(fit_id,target)，核验完整固定父链后按原导出边界原子替换文件并返回同内容的独立文档。文档为 {format:"correlated-review-v1",fit:固定拟合,cohort:完整固定队列}；其中所有 Fraction 编为约分有理字符串，整数计数仍为整数。训练目标及原始特征可由 fit.scope.revision_ids 在 cohort.rows 中精确复算。损坏内容或父链抛 ArchiveIntegrityError，未知 ID 或错误模型种类抛 ValueError，输出失败保留 OSError、先前文件及归档，不残留自有暂存。
新增数据必须能与旧成果在同一归档中共存，旧库打开不清空、不重写旧成果。内部编码方案自定。

## 手算调用与核验入口

correlated_examples.py 仅提供输入、手算预期和公开 API 调用，不包含分解或求解实现。
- x=[-1,1]、y=[0,2]、C 全 1、lambda=1：β=[1,1]、残差全 0，loss=0、penalty=objective=1。若忽略支持约束，不能得到该结果。
- 同样数据 C 为零矩阵仍须得到 β=[1,1]。若 x 恒为 7、y=[0,2]、C 全 1，则无可行拟合，拒绝且归档不变。
- x 恒为 7、y=[0,2]、C=diag(1,4)、lambda=1：β=[2/5,0]，r=[-2/5,8/5]，loss=objective=2/5，penalty=0。两新行 B 都为 [1,0]、D 全 1，均值均为 0，完整协方差全 0；B 全 0、D 全 1，则均值均为 2/5，完整协方差全 1。
- C 全 1 时，单新行 B=[1,0]、D=[[1]] 必须拒绝：形式 Schur 补为 3/4，但联合矩阵不是半正定。

实现后运行：
    python3.11 -B correlated_examples.py verify
该入口先执行全部原回归，再核验相关链路；遇失败保留非零退出码。调用夹具不是全部边界的替代品，按上述公开规则补独立核账，说明支持约束、训练轴及预测消费的交接不变量。演示入口：
    python3.11 -B correlated_examples.py demo
演示实际接纳、相关拟合、条件预测、更正、导出和关闭重开，并展示真实均值和完整协方差。沿现有 .pointwise-runtime.json 接通联合核验与该演示，保留原准备方式。不要修改原测试。
