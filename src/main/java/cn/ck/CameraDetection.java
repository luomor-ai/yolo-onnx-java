package cn.ck;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import cn.ck.config.ODConfig;
import cn.ck.domain.ODResult;
import cn.ck.utils.ImageUtil;
import cn.ck.utils.Letterbox;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.nio.FloatBuffer;
import java.util.HashMap;

/**
 * 摄像头识别，仅支持运行在有摄像头的电脑上
 */

public class CameraDetection {




    public static void main(String[] args) throws OrtException {

        //System.load(ClassLoader.getSystemResource("lib/opencv_java460.dll").getPath());
        nu.pattern.OpenCV.loadLocally();

        //linux和苹果系统需要注释这一行，如果仅打开摄像头预览，这一行没有用，可以删除，如果rtmp或者rtsp等等这一样有用，也可以用pom依赖代替
        String OS = System.getProperty("os.name").toLowerCase();
        if (OS.contains("win")) {
            System.load(ClassLoader.getSystemResource("lib/opencv_videoio_ffmpeg460_64.dll").getPath());
        }
        String model_path = "src\\main\\resources\\model\\yolov7-tiny.onnx";

        String[] labels = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
                "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"};

        // 加载ONNX模型
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 使用gpu,需要本机按钻过cuda，并修改pom.xml，不安装也能运行本程序
        // sessionOptions.addCUDA(0);

        OrtSession session = environment.createSession(model_path, sessionOptions);
        // 输出基本信息
        session.getInputInfo().keySet().forEach(x -> {
            try {
                System.out.println("input name = " + x);
                System.out.println(session.getInputInfo().get(x).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });

        // 加载标签及颜色
        ODConfig odConfig = new ODConfig();
        VideoCapture video = new VideoCapture();

        // 也可以设置为rtmp或者rtsp视频流：video.open("rtmp://192.168.1.100/live/test"), 海康，大华，录像机等等
        // video.open("rtsp://192.168.1.100/live/test")
        // 也可以静态视频文件：video.open("c://abc/123.mp4")  flv 等
        // 不持支h265视频编码，如果无法播放或者程序卡住，请修改视频编码格式
        video.open(0);  //获取电脑上第0个摄像头

        //可以把识别后的视频在通过rtmp转发到其他流媒体服务器，就可以远程预览视频后视频，需要使用ffmpeg将连续图片合成flv 等等，很简单。
        if (!video.isOpened()) {
            System.err.println("打开视频流失败，请先用vlc软件测试链接是否可以播放！");
        }

        // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min((int)video.get(Videoio.CAP_PROP_FRAME_WIDTH), (int)video.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        int thickness = minDwDh / ODConfig.lineThicknessRatio;
        double fontSize = minDwDh / ODConfig.fontSizeRatio;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        Mat img = new Mat();

        // 跳帧检测，一般设置为3，毫秒内视频画面变化是不大的，快了无意义，反而浪费性能
        int detect_skip = 3;

        // 跳帧计数
        int detect_skip_index = 1;

         // 最新一帧也就是上一帧推理结果
        float[][] outputData   = null;

        //当前最新一帧。上一帧也可以暂存一下
        Mat image;

        Letterbox letterbox = new Letterbox();
        OnnxTensor tensor;
        // 使用多线程和GPU可以提升帧率，一个线程拉流，一个线程模型推理，中间通过变量或者队列交换数据,代码示例仅仅使用单线程
        while (video.read(img)) {
            if ((detect_skip_index % detect_skip == 0) || outputData == null){
                image = img.clone();
                image = letterbox.letterbox(image);
                Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

                image.convertTo(image, CvType.CV_32FC1, 1. / 255);
                float[] whc = new float[3 * 640 * 640];
                image.get(0, 0, whc);
                float[] chw = ImageUtil.whc2cwh(whc);

                detect_skip_index = 1;

                FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
                tensor = OnnxTensor.createTensor(environment, inputBuffer, new long[]{1, 3, 640, 640});

                HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
                stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);

                // 运行推理
                // 模型推理本质是多维矩阵运算，而GPU是专门用于矩阵运算，占用率低，如果使用cpu也可以运行，可能占用率100%属于正常现象，不必纠结。
                OrtSession.Result output = session.run(stringOnnxTensorHashMap);

                // 得到结果,缓存结果
                outputData = (float[][]) output.get(0).getValue();
            }else{
                detect_skip_index = detect_skip_index + 1;
            }

            for(float[] x : outputData){

                ODResult odResult = new ODResult(x);
                // 业务逻辑写在这里，注释下面代码，增加自己的代码，根据返回识别到的目标类型，编写告警逻辑。等等

                // 画框
                Point topLeft = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio());
                Point bottomRight = new Point((odResult.getX1() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY1() - letterbox.getDh()) / letterbox.getRatio());
                Scalar color = new Scalar(odConfig.getOtherColor(odResult.getClsId()));

                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                String boxName = labels[odResult.getClsId()];
                Point boxNameLoc = new Point((odResult.getX0() - letterbox.getDw()) / letterbox.getRatio(), (odResult.getY0() - letterbox.getDh()) / letterbox.getRatio() - 3);

                // 也可以二次往视频画面上叠加其他文字或者数据，比如物联网设备数据等等
                Imgproc.putText(img, boxName, boxNameLoc, fontFace, 0.7, color, thickness);
                System.out.println(odResult+"   "+ boxName);

            }
            //服务器部署：由于服务器没有桌面，所以无法弹出画面预览，主要注释一下代码
            HighGui.imshow("result", img);

            // 多次按任意按键关闭弹窗画面，结束程序
            if(HighGui.waitKey(1) != -1){
                break;
            }
        }

        HighGui.destroyAllWindows();
        video.release();
        System.exit(0);

    }
}


