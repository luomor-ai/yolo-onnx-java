package cn.ck;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import cn.ck.config.ODConfig;
import cn.ck.domain.ODResult;
import cn.ck.utils.Letterbox;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.net.URL;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * 摄像头识别，仅支持运行在有摄像头的电脑上
 */

public class CameraDetection {

    static {
        // 加载opencv动态库，仅能在windows中运行，如果在linux中运行，需要加载linux动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java460.dll");
        System.load(url.getPath());
    }

    public static void main(String[] args) throws OrtException {

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

        // 使用gpu,需要本机按钻过cuda，并修改pom.xml
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
        VideoCapture camera = new VideoCapture();

        // 也可以设置为rtmp或者rtsp视频流：camera.open("rtmp://192.168.1.100/live/test""),camera.open("rtsp://192.168.1.100/live/test"")
        camera.open(0);

        if (!camera.isOpened()) {
            System.err.println("摄像头打开失败");
        }


        // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min((int)camera.get(Videoio.CAP_PROP_FRAME_WIDTH), (int)camera.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        int thickness = minDwDh / ODConfig.lineThicknessRatio;
        double fontSize = minDwDh / ODConfig.fontSizeRatio;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        Mat img = new Mat();

        int detect_skip = 2;

        // 跳帧计数
        int detect_skip_index = 1;

         // 最新一帧也就是上一帧推理结果
        float[][] outputData   = null;

        double ratio = 0.0d;
        double dw  = 0.0d;
        double dh = 0.0d;

        while (camera.read(img)) {


            if ((detect_skip_index % detect_skip == 0) || outputData == null){
                detect_skip_index = 1;

                Mat image = img.clone();
                Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

                // 更改 image 尺寸
                Letterbox letterbox = new Letterbox();
                image = letterbox.letterbox(image);

                ratio = letterbox.getRatio();
                dw = letterbox.getDw();
                dh = letterbox.getDh();
                int rows = letterbox.getHeight();
                int cols = letterbox.getWidth();
                int channels = image.channels();

                // 将Mat对象的像素值赋值给Float[]对象
                float[] pixels = new float[channels * rows * cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        double[] pixel = image.get(j, i);
                        for (int k = 0; k < channels; k++) {
                            // 这样设置相当于同时做了image.transpose((2, 0, 1))操作
                            pixels[rows * cols * k + j * cols + i] = (float) pixel[k] / 255.0f;
                        }
                    }
                }

                // 创建OnnxTensor对象
                long[] shape = {1L, (long) channels, (long) rows, (long) cols};
                OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);
                HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
                stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);

                // 运行推理
                OrtSession.Result output = session.run(stringOnnxTensorHashMap);


                // 得到结果
                outputData = (float[][]) output.get(0).getValue();
            }else{
                detect_skip_index = detect_skip_index + 1;
            }

            for(float[] x : outputData){

                ODResult odResult = new ODResult(x);
                System.out.println(odResult);

                // 画框
                Point topLeft = new Point((odResult.getX0() - dw) / ratio, (odResult.getY0() - dh) / ratio);
                Point bottomRight = new Point((odResult.getX1() - dw) / ratio, (odResult.getY1() - dh) / ratio);
                Scalar color = new Scalar(odConfig.getOtherColor(odResult.getClsId()));

                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                String boxName = labels[odResult.getClsId()];
                Point boxNameLoc = new Point((odResult.getX0() - dw) / ratio, (odResult.getY0() - dh) / ratio - 3);

                Imgproc.putText(img, boxName, boxNameLoc, fontFace, fontSize, color, thickness);

            }


            HighGui.imshow("result", img);

            // 按任意按键关闭弹窗画面，结束程序
            HighGui.waitKey(1);


        }

        HighGui.destroyAllWindows();
        System.exit(0);


    }



}
