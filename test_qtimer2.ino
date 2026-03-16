#include <FreqMeasureMulti.h>

FreqMeasureMulti enc1;
FreqMeasureMulti enc2;
FreqMeasureMulti enc3;
FreqMeasureMulti enc4;
FreqMeasureMulti enc5;
FreqMeasureMulti enc6;

void setup() {
    enc1.begin(14, FREQMEASUREMULTI_MARK_ONLY);
    enc2.begin(15, FREQMEASUREMULTI_MARK_ONLY);
    enc3.begin(22, FREQMEASUREMULTI_MARK_ONLY);
    enc4.begin(23, FREQMEASUREMULTI_MARK_ONLY);
    enc5.begin(18, FREQMEASUREMULTI_MARK_ONLY);
    enc6.begin(19, FREQMEASUREMULTI_MARK_ONLY);
}

void loop() {
    if (enc1.available()) {
        float freq = enc1.countToNanoseconds(enc1.read()) / 1000.0f;
    }
}
