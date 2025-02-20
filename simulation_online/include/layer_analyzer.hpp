#ifndef LAYER_ANALYZER_H
#define LAYER_ANALYZER_H

#include <stdint.h>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include "defs.hpp"

class layer_analyzer {
private:
    std::string anal_layer_name;
    uint32_t    anal_loads;
    uint32_t    anal_stores;
    uint32_t    anal_additions;
    uint32_t    anal_multiplications;
    uint32_t    anal_divisions;
    double      anal_intensity;
    bool        print_once;
public:
    layer_analyzer(std::string layer_name);
    ~layer_analyzer();

    void print_stats();
    void print_stats_raw();
    void print_stats_colnames();
    void calc_intensity();

    void incr_loads();
    void incr_stores();
    void incr_additions();
    void incr_multiplications();
    void incr_divisions();
};


#endif
