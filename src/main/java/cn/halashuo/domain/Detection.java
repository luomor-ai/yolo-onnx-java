package cn.halashuo.domain;

public class Detection{

    public String label;

    public float[] bbox;

    public float confidence;


    public Detection(String label, float[] bbox, float confidence){

        this.label = label;
        this.bbox = bbox;
        this.confidence = confidence;
    }

    public Detection(){

    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public float[] getBbox() {
        return bbox;
    }

    public void setBbox(float[] bbox) {
    }


    @Override
    public String toString() {
        return "  label="+label +
                " \t x0="+bbox[0] +
                " \t y0="+bbox[1] +
                " \t x1="+bbox[2] +
                " \t y1="+bbox[3] +
                " \t score="+confidence;
    }
}
