#ifndef JOB_SHOP_ALGORITHM_H
#define JOB_SHOP_ALGORITHM_H

#include "job_shop_environment.h"
#include <functional>

class JobShopAlgorithm {
public:
    virtual ~JobShopAlgorithm() = default;
    virtual void train(int numEpisodes, const std::function<void(int)>& episodeCallback = [](int){}) = 0;
    virtual void printBestSchedule() = 0;
    virtual void saveBestScheduleToFile(const std::string &filename) = 0;
};

#endif // JOB_SHOP_ALGORITHM_H