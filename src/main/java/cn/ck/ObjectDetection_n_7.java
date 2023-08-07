package cn.ck;

import ai.onnxruntime.*;
import cn.ck.domain.ODResult;
import cn.ck.config.ODConfig;
import cn.ck.utils.Letterbox;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * 主文件可直接运行，仅针对yolov7目标检测,可用于http接口
 *
 * 作者：常康
 */
public class ObjectDetection_n_7 {

    static {
        // 加载opencv动态库，仅能在windows中运行，如果在linux中运行，需要加载linux动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java460.dll");
        System.load(url.getPath());
    }

    public static void main(String[] args) throws OrtException {

        String model_path = "src\\main\\resources\\model\\helmet_n_7.onnx";

        // 加载ONNX模型
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 使用gpu,需要本机按钻过cuda，并修改pom.xml
        // sessionOptions.addCUDA(0);

        OrtSession session = environment.createSession(model_path, sessionOptions);
        // 输出基本信息
        session.getInputInfo().keySet().forEach(x-> {
            try {
                System.out.println("input name = " + x);
                System.out.println(session.getInputInfo().get(x).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });

        // 要检测的图片所在目录
        String imagePath = "images";

        // 加载标签及颜色
        ODConfig odConfig = new ODConfig();
        Map<String, String> map = getImagePathMap(imagePath);

        for(String fileName : map.keySet()){
            String imageFilePath = map.get(fileName);
            System.out.println(imageFilePath);
            // 读取 image
            Mat img = Imgcodecs.imread(imageFilePath);
            Mat image = img.clone();
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);


            // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
            int minDwDh = Math.min(img.width(), img.height());
            int thickness = minDwDh/ODConfig.lineThicknessRatio;
            double fontSize = minDwDh/ODConfig.fontSizeRatio;
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

            // 上面代码都是初始化后静态的，不用写在循环内，所以不计算时间
            long start_time = System.currentTimeMillis();

            // 更改 image 尺寸
            Letterbox letterbox = new Letterbox();
            image = letterbox.letterbox(image);

            double ratio  = letterbox.getRatio();
            double dw = letterbox.getDw();
            double dh = letterbox.getDh();
            int rows  = letterbox.getHeight();
            int cols  = letterbox.getWidth();
            int channels = image.channels();

            // 将Mat对象的像素值赋值给Float[]对象
            float[] pixels = new float[channels * rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double[] pixel = image.get(j,i);
                    for (int k = 0; k < channels; k++) {
                        // 这样设置相当于同时做了image.transpose((2, 0, 1))操作
                        pixels[rows*cols*k+j*cols+i] = (float) pixel[k]/255.0f;
                    }
                }
            }

            // 创建OnnxTensor对象
            long[] shape = { 1L, (long)channels, (long)rows, (long)cols };
            OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);
            HashMap<String, OnnxTensor> stringOnnxTensorHashMap = new HashMap<>();
            stringOnnxTensorHashMap.put(session.getInputInfo().keySet().iterator().next(), tensor);

            // 运行推理
            OrtSession.Result output = session.run(stringOnnxTensorHashMap);


            // 得到结果
            float[][] outputData = (float[][]) output.get(0).getValue();
            Arrays.stream(outputData).iterator().forEachRemaining(x->{

                ODResult odResult = new ODResult(x);
                System.out.println(odResult);

                // 画框
                Point topLeft = new Point((odResult.getX0()-dw)/ratio, (odResult.getY0()-dh)/ratio);
                Point bottomRight = new Point((odResult.getX1()-dw)/ratio, (odResult.getY1()-dh)/ratio);
                Scalar color = new Scalar(odConfig.getColor(odResult.getClsId()));

                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                String boxName = odConfig.getName(odResult.getClsId());
                Point boxNameLoc = new Point((odResult.getX0()-dw)/ratio, (odResult.getY0()-dh)/ratio-3);

                Imgproc.putText(img, boxName, boxNameLoc, fontFace, fontSize, color, thickness);
            });
            System.out.printf("time：%d ms.", (System.currentTimeMillis() - start_time));
            System.out.println();

            // 保存图像到同级目录
            // Imgcodecs.imwrite(ODConfig.savePicPath, img);
            // 弹窗展示图像

            HighGui.imshow("result", img);

            // 按任意按键关闭弹窗画面，结束程序
            HighGui.waitKey();

        }
        HighGui.destroyAllWindows();
        System.exit(0);
    }

    public static Map<String, String> getImagePathMap(String imagePath){
        Map<String, String> map = new TreeMap<>();
        File file = new File(imagePath);
        if(file.isFile()){
            map.put(file.getName(), file.getAbsolutePath());
        }else if(file.isDirectory()){
            for(File tmpFile : Objects.requireNonNull(file.listFiles())){
                map.putAll(getImagePathMap(tmpFile.getPath()));
            }
        }
        return map;
    }
}
