## 先完整阅读文档！！！
## 项目由来
1.  在调用深度学习训练好的AI模型时，如果使用python调用非常简单，甚至不用编写代码，大部分深度学习框架就是python编写的，自带有推理逻辑文件和方法
2.  但是不是每个同学都会python,不是每个项目都是python语言开发，不是每个岗位都会深度学习
3.  由于大部分服务器项目还是由java语言居多，java方向的开发者也多，由于本人找遍全网也没有找到java调用AI模型的例子，
4.  所以特意编写一个java调用AI模型的方法(全网应该就这一份)。思路是通用的，只需要替换不同的模型即可达到不同效果
5.   **不懂项目有什么用作？不知道用在什么地方？没关系，先下载运行看效果后立马就明白了！** 

---

## 紧接着下载运行看效果再研究代码，别忘记点star
1.  下载代码可直接运行主文件：`ObjectDetection_1_25200_n.java` , `ObjectDetection_n_7.java`,`ObjectDetection_1_n_8400.java` 都 **可以直接运行不会报错** 
2.  `CameraDetection.java`，是实时视频流识别检测，也可直接运行（ **仅支持有摄像头的电脑或笔记本** ），三个文件完全独立，不互相依赖，如果有GPU帧率会更高，需要开启调用GPU
3.  多个主文件是为了支持不用网络结构的模型，即使是`onnx`模型，输出的结果参数也不一样，目前支持三种结构，下面有讲解
4.  可以封装为`http` `controller` `api`接口，也可以结合摄像头实时分析视频流，进行识别后预览和告警
5.  支持`yolov7` , `yolov5`和`yolov8`,`paddlepaddle`后处理稍微改一下也可以支持, **代码中自带的onnx模型仅仅为了演示，准确率非常低，实际应用需要自己训练** 
6.  训练出来的模型成为基础模型，可以用于测试。生产环境的模型需要经过模型压缩，量化，剪枝，蒸馏，才可以使用(当然这不是java开发者的工作)。会提升视频华民啊帧率达到60-120帧左右。点击查看：[百度压缩模型工具](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3949129)，[基础概念](https://zhuanlan.zhihu.com/p/138059904)，[参考文章](https://zhuanlan.zhihu.com/p/430910227)
6.  替换`model`目录下的onnx模型文件，可以识别检测任何物体(烟火，跌倒，抽烟，安全帽，口罩，人，打架，计数，攀爬，垃圾，开关，状态，分类，等等)，有模型即可
7.  模型不是onnx格式怎么办？不要紧张，主流模型都可以转为onnx格式。怎么转？看完文档就知道了！
---

## ObjectDetection_1_25200_n.java
 - `yolov5`
 - **85**：每一行`85`个数值，`5`个center_x,center_y, width, height，score ，`80`个标签类别得分(不一定是80要看模型标签数量)
 - **25200**：三个尺度上的预测框总和 `( 80∗80∗3 + 40∗40∗3 + 20∗20∗3 )`，每个网格三个预测框，后续需要`非极大值抑制NMS`处理
 - **1**：没有批量预测推理，即每次输入推理一张图片
![输入图片说明](https://foruda.gitee.com/images/1690944300550600655/cdf2a2cb_1451768.png "屏幕截图")

---

## ObjectDetection_n_7.java
 - `yolov7`
 - **Concatoutput_dim_0** ：变量，表示当前图像中预测目标的数量，
 - **7**：表示每个目标的七个参数：`batch_id，x0，y0，x1，y1，cls_id，score`
![输入图片说明](https://foruda.gitee.com/images/1690944320288742664/eb1cb2d9_1451768.png "屏幕截图")

---

## ObjectDetection_1_n_8400.java
 - `yolov8`                                                                
![输入图片说明](https://foruda.gitee.com/images/1692002728787198481/9b1b9a16_1451768.png "20230814164509.png")

---
## 暂不直接支持输出结果是三个数组参数的以下模型
- 但是这种结构模型可以导出为`[1,25200,85]`或`[n,7]`输出结构，然后就可以使用已有代码调用。
-  **yolov5** ：导出onnx时增加参数  `inplace=True,simplify=True`(ObjectDetection_1_25200_n.java)
-  **yolov7** ：导出onnx时增加参数  `grid=True,simplify=True`(ObjectDetection_1_25200_n.java) 或者 `grid=True,simplify=True,end2end=True,include-nms=True`(ObjectDetection_n_7.java)
![输入图片说明](https://foruda.gitee.com/images/1691765789379434579/3c314f1c_1451768.png "屏幕截图")
![输入图片说明](https://foruda.gitee.com/images/1691766358544706096/1136ee49_1451768.png "屏幕截图")

---

## ONNX
Open Neural Network Exchange（ONNX，开放神经网络交换）格式，是一个用于表示深度学习模型的标准，可使模型在不同框架之间进行转移.
是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch,TensorFlow,PaddlePaddle,MXNet）可以采用相同格式存储模型数据并交互。 ONNX的规范及代码主要由微软，亚马逊 ，Facebook 和 IBM 等公司共同开发，以开放源代码的方式托管在Github.

## 效果预览
![输入图片说明](https://foruda.gitee.com/images/1691564940451414777/1d31975d_1451768.png)

### 扫码拉入群(备注：onnx)
![输入图片说明](https://foruda.gitee.com/images/1692414289523254420/c430b1ce_1451768.jpeg "微信图片_20230819110428.jpg")

## 有用链接
- https://blog.csdn.net/changzengli/article/details/129182528
- https://blog.csdn.net/xqtt29/article/details/110918397
- https://blog.csdn.net/changzengli/article/details/127904594
- 使用封装后的javacpp中的javacv 和 ffmpeg 也可以


