## 不点star不给解答问题
1.  下载代码可直接运行主文件：`ObjectDetection_1_25200_n.java` 或者 `ObjectDetection_n_7.java` 都可以直接运行
2.  两个可以运行的主文件是为了支持不用网络结构的模型，即使是`onnx`模型，输出的结果参数也不一样，支持以下两种结构
3.  目前代码仅支持`window`s系统，`linux`需要替换`opencv`的`dll`文件为`so`文件
4.  可以封装为HTTP controller API接口
5.  支持`yolov7` 和 `yolov5`，代码中自带的onnx模型仅仅为了演示，准确率非常低，实际应用需要自己训练
6.  替换`model`目录下的onnx模型文件，可以识别检测任何物体(烟火，跌倒，抽烟，安全帽，口罩，人，等等)，有模型即可

---

## ObjectDetection_1_25200_n.java
 - **85**：每一行`85`个数值，`5`个center_x,center_y, width, height，score ，`80`个标签类别得分(不一定是80要看模型标签数量)
 - **25200**：三个尺度上的预测框总和 `( 80∗80∗3 + 40∗40∗3 + 20∗20∗3 )`，每个网格三个预测框，后续需要`非极大值抑制NMS`处理
 - **1**：没有批量预测推理，即每次输入推理一张图片
![输入图片说明](https://foruda.gitee.com/images/1690944300550600655/cdf2a2cb_1451768.png "屏幕截图")
---

## ObjectDetection_n_7.java
 - **Concatoutput_dim_0** ：变量，表示当前图像中预测目标的数量，
 - **7**：表示每个目标的七个参数：`batch_id，x0，y0，x1，y1，cls_id，score`
![输入图片说明](https://foruda.gitee.com/images/1690944320288742664/eb1cb2d9_1451768.png "屏幕截图")

## 暂不支持返回三个数组参数的模型
