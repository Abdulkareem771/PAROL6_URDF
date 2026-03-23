#include <FreqMeasureMulti.h>

FreqMeasureMulti freq1;
FreqMeasureMulti freq2;
FreqMeasureMulti freq3;
FreqMeasureMulti freq4;
FreqMeasureMulti freq5;
FreqMeasureMulti freq6;

void setup() {
  bool b1 = freq1.begin(14, FREQMEASUREMULTI_SPACE_ONLY);
  bool b2 = freq2.begin(15, FREQMEASUREMULTI_SPACE_ONLY);
  bool b3 = freq3.begin(16, FREQMEASUREMULTI_SPACE_ONLY);
  bool b4 = freq4.begin(17, FREQMEASUREMULTI_SPACE_ONLY);
  bool b5 = freq5.begin(18, FREQMEASUREMULTI_SPACE_ONLY);
  bool b6 = freq6.begin(19, FREQMEASUREMULTI_SPACE_ONLY);
  
}

void loop() {}
