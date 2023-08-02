package cn.halashuo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import cn.halashuo.config.ODConfig;
import cn.halashuo.domain.Detection;
import cn.halashuo.utils.Letterbox;
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
import java.util.stream.Collectors;

/**
 * 主文件可直接运行，仅针对yolov5目标检测,可用于http接口
 *
 * 作者：常康
 */
public class ObjectDetection_1_25200_n {

    static {
        // 加载opencv动态库，仅能在windows中运行，如果在linux中运行，需要加载linux动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java460.dll");
        System.load(url.getPath());
    }

    public static void main(String[] args) throws OrtException {

        String model_path = "src\\main\\resources\\model\\helmet_1_25200_n.onnx";

        float confThreshold = 0.35F;

        float nmsThreshold = 0.55F;

        String[] labels = {"no_helmet", "helmet"};

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



           // img.convertTo(img, CvType.CV_32FC1, 1. / 255);
           // float[] whc = new float[NUM_INPUT_ELEMENTS];
           // img.get(0, 0, whc);
           // float[] chw = ImageUtil.whc2cwh(whc);
           // FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
           // inputTensor = OnnxTensor.createTensor(this.env, inputBuffer, INPUT_SHAPE);

            // 在这里先定义下框的粗细、字的大小、字的类型、字的颜色(按比例设置大小粗细比较好一些)
            int minDwDh = Math.min(img.width(), img.height());
            int thickness = minDwDh/ODConfig.lineThicknessRatio;
            double fontSize = minDwDh/ODConfig.fontSizeRatio;
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
            Scalar fontColor = new Scalar(255, 255, 255);
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
            float[][] outputData = ((float[][][])output.get(0).getValue())[0];
            Map<Integer, List<float[]>> class2Bbox = new HashMap<>();
            for (float[] bbox : outputData) {

                // center_x,center_y, width, height，score
                float score = bbox[4];
                if (score < confThreshold) continue;

                // 获取标签
                float[] conditionalProbabilities = Arrays.copyOfRange(bbox, 5, bbox.length);
                int label = argmax(conditionalProbabilities);

                // xywh to (x1, y1, x2, y2)
                xywh2xyxy(bbox);

                // 去除无效结果
                if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) continue;

                class2Bbox.putIfAbsent(label, new ArrayList<>());
                class2Bbox.get(label).add(bbox);
            }

            List<Detection> detections = new ArrayList<>();
            for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {

                List<float[]> bboxes = entry.getValue();
                bboxes = nonMaxSuppression(bboxes, nmsThreshold);
                for (float[] bbox : bboxes) {
                    String labelString = labels[entry.getKey()];
                    detections.add(new Detection(labelString, Arrays.copyOfRange(bbox, 0, 4), bbox[4]));
                }
            }

            for (Detection detection : detections) {
                float[] bbox = detection.getBbox();
                System.out.println(detection.toString());
                // 画框
                Point topLeft = new Point((bbox[0]-dw)/ratio, (bbox[1]-dh)/ratio);
                Point bottomRight = new Point((bbox[2]-dw)/ratio, (bbox[3]-dh)/ratio);
                Scalar color = new Scalar(odConfig.getColor(1));
                Imgproc.rectangle(img, topLeft, bottomRight, color, thickness);
                // 框上写文字
                Point boxNameLoc = new Point((bbox[0]-dw)/ratio, (bbox[1]-dh)/ratio-3);

                Imgproc.putText(img, detection.getLabel(), boxNameLoc, fontFace, fontSize, fontColor, thickness);
            }
            System.out.printf("time：%d ms.", (System.currentTimeMillis() - start_time));
            System.out.println();

            // 保存图像到同级目录
            // Imgcodecs.imwrite(ODConfig.savePicPath, img);
            // 弹窗展示图像
            HighGui.imshow("Display Image", img);
            // 按任意按键关闭弹窗画面，结束程序
            HighGui.waitKey();
        }
        HighGui.destroyAllWindows();
        System.exit(0);

    }



    public static void scaleCoords(float[] bbox, float orgW, float orgH, float padW, float padH, float gain) {
        // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
        bbox[0] = Math.max(0, Math.min(orgW - 1, (bbox[0] - padW) / gain));
        bbox[1] = Math.max(0, Math.min(orgH - 1, (bbox[1] - padH) / gain));
        bbox[2] = Math.max(0, Math.min(orgW - 1, (bbox[2] - padW) / gain));
        bbox[3] = Math.max(0, Math.min(orgH - 1, (bbox[3] - padH) / gain));
    }
    public static void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f;
        bbox[1] = y - h * 0.5f;
        bbox[2] = x + w * 0.5f;
        bbox[3] = y + h * 0.5f;
    }

    public static List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {


        List<float[]> bestBboxes = new ArrayList<>();


        bboxes.sort(Comparator.comparing(a -> a[4]));


        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1);
            bestBboxes.add(bestBbox);
            bboxes = bboxes.stream().filter(a -> computeIOU(a, bestBbox) < iouThreshold).collect(Collectors.toList());
        }

        return bestBboxes;
    }

    public static float computeIOU(float[] box1, float[] box2) {

        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float interArea = Math.max(right - left, 0) * Math.max(bottom - top, 0);
        float unionArea = area1 + area2 - interArea;
        return Math.max(interArea / unionArea, 1e-8f);

    }

    public static int argmax(float[] a) {
        float re = -Float.MAX_VALUE;
        int arg = -1;
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= re) {
                re = a[i];
                arg = i;
            }
        }
        return arg;
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
