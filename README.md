# 海岬检测中心：数值预测归档

Python 3.11 标准库离线应用。分析员接纳仪器测量、更正和撤回，固定明确修订的分析队列；检测岗位使用固定模型预测浓度；质控员重开归档并导出原始训练依据和留出证据。数据库和导出文件由调用方放在源码目录之外。没有安装依赖。

## 运行

在项目根目录运行：

    python3.11 -B -m unittest discover -s tests
    python3.11 -B -m delivery.workflows

演示使用独立临时目录，实际执行接入、固定、拟合、留出评估、预测、导出和重开核对，退出后清理。指定持久输出：

    python3.11 -B -m delivery.workflows --database /tmp/haijia-demo.sqlite --export /tmp/haijia-review.json

重复运行会重放样例版本并返回相同成果；不会清空用户更正，也不会把当前头倒退。修改样例内容时必须提高对应行的源版本，否则同版本冲突明确失败。不要把 SQLite 文件或导出文件放入仓库。

## 一天的工作

1. 打开归档，检查模式版本并保留已有成果。
2. 接纳离线事件批次，整批合法后提交修订和当前头。
3. 用明确修订 ID 和特征顺序固定队列，供训练和复核使用。
4. 选择训练设备，包含这些设备在队列内的全部重复行。
5. 只用训练行标准化并求解正则线性模型，归档诊断。
6. 选择不交的留出设备，计算逐行误差并发布完整预测包。
7. 检测岗位按包内位置顺序输入新数值行，取得浓度预测。
8. 质控员关闭重开并导出固定父链、训练投影和留出证据。

## 公开 API

    from delivery.app import open_archive

    with open_archive("/tmp/haijia.sqlite") as app:
        revisions = app.apply_measurements([
            {"row_id": "m1", "device_id": "A", "version": 1,
             "features": {"x": -1}, "target": 9},
            {"row_id": "m2", "device_id": "A", "version": 1,
             "features": {"x": 1}, "target": 11},
            {"row_id": "m3", "device_id": "B", "version": 1,
             "features": {"x": 3}, "target": 13}
        ])
        cohort_id = app.freeze_cohort(
            [r["revision_id"] for r in revisions], ["x"])
        fit = app.fit_numeric(cohort_id, {"A"}, "1")
        package = app.evaluate_holdout(fit["fit_id"], {"B"})
        values = app.predict_numeric(package["package_id"], [[5]])
        graph = app.read_package(package["package_id"])
        evidence = app.export_package(package["package_id"], "/tmp/haijia-evidence.json")

apply_measurements 返回与输入位置对应的修订字典列表；freeze_cohort 返回 ID；
fit_numeric 返回含 fit_id 的固定拟合字典；evaluate_holdout 返回含 package_id、
逐行 holdout 和 error={sse,n,mse} 的包字典。read_package 返回
{package,fit,cohort} 完整独立父图。另有 read_cohort 和 read_fit 供复核。
export_package 原子写入单份 JSON 并返回同内容的独立文档。close 幂等，
上下文管理器自动关闭。底层同名函数位于相应职责模块，第一参数为 ArchiveStore。
存储会话按单调用顺序使用，不提供跨线程会话共享。

事件默认 op="put"；row_id、device_id、特征名为非空字符串，version 为正整数。
数值接受 int、有限 float、Fraction 或精确有理字符串，如 "1/3"；
拒绝 bool、NaN、无穷和不能表示为有限浮点的数。归档按约分有理字符串保存
源数值，重放比较精确内容，不用计算容差。float 按十进制 str(value) 规范化。
只支持数值特征。features 是非空映射，target 必填，未知事件字段拒绝。

更正提交更高版本 put。撤回可只提交
{"row_id":"m1","version":2,"op":"withdraw"}，复制原载荷形成新修订；
也可同时显式给出 device_id、features、target 三项，不能只给其中部分。
未知对象不可撤回。历史版本相同内容可重放，未接纳的倒序版本和同版本冲突
整批拒绝，合法前缀也不提交。空事件批次返回 []。
固定队列必须非空、成员唯一、修订存在且未撤回；特征顺序非空唯一，
所有成员均须提供这些列。历史有效修订可明确选择，后续撤回不影响旧队列。

训练和留出参数都是设备 ID 的可迭代集合，不是行 ID。训练和留出均不得为空，
不得包含未知设备，二者任何交集都拒绝。留出不必覆盖所有未训练设备。
lambda_ 必须为正且可表示的有理数。预测输入按 feature_order 位置排列，
不带目标；返回输入顺序的列表，空输入返回 []，非法行整次拒绝。
所有返回嵌套对象均与归档隔离；对象修改不会改变未来预测、读取或导出。

## 数学口径和样例

训练行数 n；均值和总体标准差只使用训练行。非常量列
z=(x-mu)/sigma；常量列保存 effective scale=1，训练、留出、新行一律变换为 0。
求解 J=Σ(y-b-w·z)²/n + lambda*Σw²，截距不惩罚。
系数数组位置为 [截距, 按 feature_order 的系数]。
直接带主元求解；良态输入数值绝对误差不超过 1e-8，无随机过程。
无法保证有限计算或求解成功时拒绝，不发布默认结果。

examples/day.json 的 x 是温度偏差（摄氏度），reference 是参考通道信号（毫伏），
target 是浓度（mg/L）。训练 A、B 的 x 为 -2,-1,1,2，target=10+x，
reference=7，lambda=1；均值 [0,7]，尺度 [sqrt(2.5),1]，第二列常量。
截距 10，系数 [sqrt(2.5)/2,0]，训练 MSE=0.625，正则项=0.625，
目标函数=1.25。留出 C 的 x=4，D 三行 x=6，预测为 12,13,13,13；
SSE=31，行数=4，MSE=7.75。新行 [8,7] 预测 14 mg/L。
误差是全体留出行 SSE/n，不是设备 MSE 的平均值。

独立正交夹具为 (-1,-1),(-1,1),(1,-1),(1,1)，目标 7,9,11,13，
lambda=1；截距 10，系数 [1,0.5]，训练 MSE 和正则项各 1.25，
目标函数 2.5。tests/test_daily_fit.py 独立固定这些预期。

## 状态、失败与代码归属

measurements 的四个模块负责规范化、不可变修订、批次规划与原子接纳；
cohorts 的四个模块负责明确选样、快照、身份和完整设备范围；
fitting 的四个模块负责标准化、目标方程、真实求解与诊断；
prediction 的四个模块负责固定包、执行、留出和行误差；
archive 的四个模块负责模式、SQLite 事务、有序子记录、父关系和固定读取；
delivery 的四个模块负责门面、每日流程、复核清单和原子导出。

身份由带版本规范 JSON 的 SHA-256 确定，包含有序成员、特征、数值和明确父关系。
SQLite 启用外键；模式版本使用 PRAGMA user_version=1，初始化幂等。
未知版本和未标版本的非空库拒绝。队列、拟合、包只有完整发布状态对外可见，
候选、标准化、求解和评估中间状态仅在调用内存中存在。
头、有序子记录和父链接在同一事务中提交，重试和成功重放不重复创建成果。
读取核对内容身份、类型、父链、成员和子记录完整性，返回固定证据，不重算模型。
导出先核验父图，再同目录暂存、flush/fsync、os.replace；失败清理自有暂存，
保留归档和先前目标文件。输出父目录必须已经存在。

输入、版本、设备和数值错误抛 ValueError；未知成果也抛 ValueError；
求解失败抛 NumericSolveError（ValueError 子类）；损坏父链或内容抛
ArchiveIntegrityError；输出文件失败保留原 OSError。
publication_hook(stage,artifact_id,kind) 可通过 open_archive 参数传入，
stage 为 after_header、after_children、before_commit，kind 为 cohort、fit、package；
回归用该公开端口抛 PublicationFailure 核验回滚，默认无钩子。
fit_numeric 的 solver(matrix,rhs) 关键字及 open_archive 的 solver 参数提供
求解依赖替换；默认始终调用 fitting/solver.py 的真实求解器。

完整回归和操作说明见 RUNBOOK.md；运行清单见 .pointwise-runtime.json。
