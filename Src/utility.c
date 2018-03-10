

#include "utility.h"


uint32_t index_of_most_probable(float32_t probabilities[10]){
  uint32_t index_of_max = 0;
  for(uint32_t i=0; i<10;i++){
    if (probabilities[i] > probabilities[index_of_max]) index_of_max = i;
  }

  return index_of_max;
}

