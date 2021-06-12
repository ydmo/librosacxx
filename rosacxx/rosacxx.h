#ifndef ROSACXX_ROSACXX_H
#define ROSACXX_ROSACXX_H

#include <cfloat>
#include <memory>

namespace rosacxx {

double getTickFrequency(void);
double getTickCount(void);

#define LOGTIC(event) double tic_##event = rosacxx::getTickCount();
#define LOGTOC(event) printf("[ROSACXX] Event[%s] cost %f ms\n", #event, ( rosacxx::getTickCount() - tic_##event ) / rosacxx::getTickFrequency() * 1e3) ;

class TimeMetrics {
public:
    typedef std::shared_ptr<TimeMetrics> Ptr;
    TimeMetrics() = default;
    ~TimeMetrics() = default;

    void clear() {
        _tic = 0;
        _sum = 0;
        _cnt = 0;
        _min = FLT_MAX;
        _max = FLT_MIN;
    }

    double sum() const {
        return _sum;
    }

    size_t cnt() const {
        return _cnt;
    }

    double max() const {
        return _max;
    }

    double min() const {
        return _min;
    }

    double avg() const {
        return _sum / _cnt;
    }

    double feq() const {
        return getTickFrequency();
    }

    double tic() {
        _tic = getTickCount();
        return _tic;
    }

    double toc() {
        double toc = getTickCount();
        double elapse = (toc - _tic) / getTickFrequency();
        _min = std::min(_min, elapse);
        _max = std::max(_max, elapse);
        _sum += elapse;
        _cnt += 1;
        return toc;
    }

public:
    double _min = FLT_MAX;
    double _max = FLT_MIN;
    double _sum = 0;
    size_t _cnt = 0;

protected:
    double _tic = 0;
};

} // namespace rosacxx

#endif // ROSACXX_ROSACXX_H
