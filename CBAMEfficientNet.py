from mindcv.data import create_dataset, create_transforms, create_loader
import os
import mindspore as ms
from mindspore import Model, save_checkpoint, load_checkpoint,  nn
from mindspore.train.callback import Callback
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

batch_size = 64
num_workers = 16
num_classes = 31
data_dir = "./rockdataset4"

# 加载数据集
dataset_train = create_dataset(root=data_dir, split='train', num_parallel_workers=num_workers, shuffle=True)
dataset_val = create_dataset(root=data_dir, split='val', num_parallel_workers=num_workers, shuffle=True)

class_names = dataset_train.get_class_indexing()
class_names = dict(zip(class_names.values(),class_names.keys()))
print(class_names)

# 数据增强
trans = create_transforms(is_training=True, image_resize=384,scale=(0.5, 1.0), color_jitter=(0.2, 0.2, 0.2, 0.02))
trans_val = create_transforms(is_training=False, image_resize=384)

# 创建数据加载器
loader_train = create_loader(dataset=dataset_train,
                             batch_size=batch_size,
                             is_training=True,
                             num_classes=num_classes,
                             transform=trans,
                             num_parallel_workers=num_workers)

num_batches = loader_train.get_dataset_size()

# 保留空间注意力模块

class SpatialAttention(nn.Cell):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, pad_mode='same')
        self.sigmoid = nn.Sigmoid()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)  # 计算通道均值
        self.reduce_max = ops.ReduceMax(keep_dims=True)  # 计算通道最大值

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)  # 计算通道均值
        max_out = self.reduce_max(x, 1)   # 计算通道最大值
        x = ops.Concat(1)((avg_out, max_out))  # 拼接后通道数变为 2
        x = self.conv1(x)  # 通过 7x7 卷积层
        return self.sigmoid(x)  # 维度仍然是 (batch, 1, H, W)

class CBAM(nn.Cell):
    def __init__(self, kernel_size=7):
        super(CBAM, self).__init__()
        self.sa = SpatialAttention(kernel_size)

    def construct(self, x):
        x = x * self.sa(x)  # 仅应用空间注意力机制
        return x

# 创建模型并添加 CBAM
from mindcv.models import create_model

class CBAMEfficientNet(nn.Cell):
    def __init__(self, base_model, num_classes, apply_softmax=False):
        super(CBAMEfficientNet, self).__init__()
        self.features = base_model.features  # EfficientNet 的特征提取部分
        self.cbam = CBAM(kernel_size=7)  # 仅使用空间注意力机制
        self.reduce_mean = ops.ReduceMean(keep_dims=True)  # 计算全局均值
        self.flatten = nn.Flatten()  # 展平
        
        # 使用 mlp_head 作为分类层
        self.classifier = base_model.mlp_head  
        self.apply_softmax = apply_softmax  # 控制是否应用 Softmax

    def construct(self, x):
        x = self.features(x)  # EfficientNet 提取特征，输出 (batch, 1280, H, W)
        x = self.cbam(x)  # 仅应用空间注意力
        x = self.reduce_mean(x, (2, 3))  # 计算全局均值，变为 (batch, 1280, 1, 1)
        x = self.flatten(x)  # 变为 (batch, 1280)
        x = self.classifier(x)  # 送入分类层

        if self.apply_softmax:
            x = ops.Squeeze()(x)
            x = ops.Softmax()(x)  # 计算概率
        return x



# 加载预训练模型
base_model = create_model(model_name='efficientnet_b1', num_classes=num_classes, pretrained=True)
network = CBAMEfficientNet(base_model, num_classes)

#ms.amp.auto_mixed_precision(network, amp_level="O2")  # "O2" 是推荐级别

# 创建损失函数
from mindcv.loss import create_loss
loss = create_loss(name='CE')

# 设置学习率策略
from mindcv.scheduler import create_scheduler
lr_scheduler = create_scheduler(
    steps_per_epoch=num_batches,
    scheduler='constant',
    lr=0.001,
    min_lr=1e-6,
    warmup_epochs=5,
)

# 设置优化器
from mindcv.optim import create_optimizer
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)

# 封装可训练或推理的实例
model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})

# 定义保存路径
ckpt_save_dir = './ckpt'
if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)

# 训练回调
class BestCheckpointCallback(Callback):
    def __init__(self, model, eval_loader, save_path, eval_interval=1):
        super(BestCheckpointCallback, self).__init__()
        self.model = model
        self.eval_loader = eval_loader
        self.save_path = save_path
        self.best_acc = 0.0
        self.best_epoch = 0
        self.eval_interval = eval_interval

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        
        if epoch % self.eval_interval == 0:
            acc = self.model.eval(self.eval_loader, dataset_sink_mode=False)
            acc_value = acc.get("accuracy", 0.0)
            print(f"Epoch {epoch}: 验证集准确率 = {acc_value:.4f}")
            
            if acc_value > self.best_acc:
                self.best_acc = acc_value
                self.best_epoch = epoch
                best_ckpt_path = os.path.join(self.save_path, "best_checkpoint.ckpt")
                save_checkpoint(self.model._network, best_ckpt_path)
                print(f"🏆 新的最佳模型已保存！Epoch {epoch}, 最高准确率: {self.best_acc:.4f}")

# 创建验证集数据加载器
loader_val = create_loader(dataset=dataset_val,
                           batch_size=batch_size,
                           is_training=False,
                           num_classes=num_classes,
                           transform=trans_val,
                           num_parallel_workers=num_workers)

best_ckpt_cb = BestCheckpointCallback(model, loader_val, ckpt_save_dir)

from mindspore import LossMonitor, TimeMonitor

best_ckpt_cb = BestCheckpointCallback(model, loader_val, ckpt_save_dir)

summary_collector = ms.SummaryCollector(summary_dir='./summary_dir/summary_02')

model.train(100, loader_train, callbacks=[LossMonitor(num_batches//5), TimeMonitor(num_batches//5), best_ckpt_cb, summary_collector], dataset_sink_mode=False)

# 评估最佳模型
best_ckpt_path = os.path.join(ckpt_save_dir, "best_checkpoint.ckpt")
load_checkpoint(best_ckpt_path, net=network)

final_acc = model.eval(loader_val, dataset_sink_mode=False)
print("🎯 最终最佳模型的验证集准确率:", final_acc)