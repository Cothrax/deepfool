# 6人无限注德州扑克AI DeepFool

### 文件说明
`train_single_gpu.py`为策略网络训练代码

`train_equity.py`为预训练代码

`data.py`为数据处理和加载代码

`data_generator.py`为预训练无history样本生成代码

`train.toml`为训练参数配置文件

`cfr_py/`为MCCFR的实现代码

`cfr_py/calculator` C++写的牌力计算器，由Swig包装，在linux可以使用，无需单独编译

`agents/`为可在`neuron_poker`环境中运行的Agent

### 预训练样本生成
#### 无history样本生成
```
python data_generator.py x # x 替换为样本标号
```
运行完后将在当前目录生成名为`data_x.pkl`的样本（包含300000样本对）。
#### 带history样本生成
TODO

### 预训练
```
python train_equity.py
```
将使用生成的样本训练`data.py`中的`PreTrain`，作为预训练模型。

### 训练
```
python train_single_gpu.py
```
将与`MYCFR`交互学习。

### 测试
首先将`agents/agent_cfr.py`、`agents/model.py`和`agents/checkpoint.pt`拷贝到`neuron_poker`环境的`agents/`目录，然后在需要使用本Agent的地方调用：
```
from agents.agent_cfr import Player as CFRPlayer
self.env.add_player(CFRPlayer(self.env))
```
其中`self.env`为`gym.make`实例化的环境。

### 库函数依赖
本仓库运行在如下环境中：
```
python==3.7
pytorch==1.6.0
numpy==1.18.1
scipy
tensorboard
tensorboardX
```

建议在linux下运行，其他系统没有测试。

### 联系
如果在运行时出现任何问题，欢迎联系：`{liangsusan18, lilongcheng18, chenyanfan18}@mails.ucas.ac.cn`。
