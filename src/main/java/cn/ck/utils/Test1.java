package cn.ck.utils;

public class Test1 {

    /*public Test(String modelPath, String labelPath, float confThreshold, float nmsThreshold, int gpuDeviceId) throws OrtException, IOException {
        super(modelPath, labelPath, confThreshold, nmsThreshold, gpuDeviceId);
    }

    @Override
    public List<Detection> run(Mat img) throws OrtException {

        float orgW = (float) img.size().width;
        float orgH = (float) img.size().height;

        float gain = Math.min((float) INPUT_SIZE / orgW, (float) INPUT_SIZE / orgH);
        float padW = (INPUT_SIZE - orgW * gain) * 0.5f;
        float padH = (INPUT_SIZE - orgH * gain) * 0.5f;


        Map<String, OnnxTensor> inputContainer = this.preprocess(img);


        float[][] predictions;

        OrtSession.Result results = this.session.run(inputContainer);
        predictions = ((float[][][]) results.get(0).getValue())[0];


        return postprocess(predictions, orgW, orgH, padW, padH, gain);
    }


    public Map<String, OnnxTensor> preprocess(Mat img) throws OrtException {


        Mat resizedImg = new Mat();
        ImageUtil.resizeWithPadding(img, resizedImg, INPUT_SIZE, INPUT_SIZE);

        // BGR -> RGB
        Imgproc.cvtColor(resizedImg, resizedImg, Imgproc.COLOR_BGR2RGB);

        Map<String, OnnxTensor> container = new HashMap<>();


        if (this.inputType.equals(OnnxJavaType.UINT8)) {
            byte[] whc = new byte[NUM_INPUT_ELEMENTS];
            resizedImg.get(0, 0, whc);
            byte[] chw = ImageUtil.whc2cwh(whc);
            ByteBuffer inputBuffer = ByteBuffer.wrap(chw);
            inputTensor = OnnxTensor.createTensor(this.env, inputBuffer, INPUT_SHAPE, this.inputType);
        } else {

            resizedImg.convertTo(resizedImg, CvType.CV_32FC1, 1. / 255);
            float[] whc = new float[NUM_INPUT_ELEMENTS];
            resizedImg.get(0, 0, whc);
            float[] chw = ImageUtil.whc2cwh(whc);
            FloatBuffer inputBuffer = FloatBuffer.wrap(chw);
            inputTensor = OnnxTensor.createTensor(this.env, inputBuffer, INPUT_SHAPE);
        }


        container.put(this.inputName, inputTensor);

        return container;
    }

    public List<Detection> postprocess(float[][] outputs, float orgW, float orgH, float padW, float padH, float gain) {


        Map<Integer, List<float[]>> class2Bbox = new HashMap<>();

        for (float[] bbox : outputs) {

            float conf = bbox[4];
            if (conf < this.confThreshold) continue;

            float[] conditionalProbabilities = Arrays.copyOfRange(bbox, 5, 85);
            int label = Yolo.argmax(conditionalProbabilities);

            // xywh to (x1, y1, x2, y2)
            xywh2xyxy(bbox);

            // skip invalid predictions
            if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) continue;

            // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
            scaleCoords(bbox, orgW, orgH, padW, padH, gain);
            class2Bbox.putIfAbsent(label, new ArrayList<>());
            class2Bbox.get(label).add(bbox);
        }


        List<Detection> detections = new ArrayList<>();
        for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {
            int label = entry.getKey();
            List<float[]> bboxes = entry.getValue();
            bboxes = nonMaxSuppression(bboxes, this.nmsThreshold);
            for (float[] bbox : bboxes) {
                String labelString = this.labelNames.get(label);
                detections.add(new Detection(labelString, Arrays.copyOfRange(bbox, 0, 4), bbox[4]));
            }
        }

        return detections;
    }*/
}
