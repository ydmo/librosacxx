#include "kiss_fft.h"

#include "ext/vamp_kiss_fft.h"
#include "ext/vamp_kiss_fftr.h"

namespace vkfft {

void fft_forward(unsigned int un, const std::complex<kiss_fft_scalar_t> *ci, std::complex<kiss_fft_scalar_t> *co) {
    int n(un);
    vamp_kiss_fft_cfg c = vamp_kiss_fft_alloc(n, false, 0, 0);
    vamp_kiss_fft_cpx *in = new vamp_kiss_fft_cpx[n];
    vamp_kiss_fft_cpx *out = new vamp_kiss_fft_cpx[n];
    for (int i = 0; i < n; ++i) {
        in[i].r = ci[i].real();
        in[i].i = ci[i].imag();
    }
    vamp_kiss_fft(c, in, out);
    for (int i = 0; i < n; ++i) {
        co[i].real(out[i].r);
        co[i].imag(out[i].i);
    }
    vamp_kiss_fft_free(c);
    delete[] in;
    delete[] out;
}

void fft_inverse(unsigned int un, const std::complex<kiss_fft_scalar_t> *ci, std::complex<kiss_fft_scalar_t> *co) {
    int n(un);
    vamp_kiss_fft_cfg c = vamp_kiss_fft_alloc(n, true, 0, 0);
    vamp_kiss_fft_cpx *in = new vamp_kiss_fft_cpx[n];
    vamp_kiss_fft_cpx *out = new vamp_kiss_fft_cpx[n];
    for (int i = 0; i < n; ++i) {
        in[i].r = ci[i].real();
        in[i].i = ci[i].imag();
    }
    vamp_kiss_fft(c, in, out);
    kiss_fft_scalar_t scale = 1.0 / double(n);
    for (int i = 0; i < n; ++i) {
        co[i].imag(out[i].r * scale);
        co[i].imag(out[i].i * scale);
    }
    vamp_kiss_fft_free(c);
    delete[] in;
    delete[] out;
}

void fft_forward(unsigned int un, const kiss_fft_scalar_t *ri, const kiss_fft_scalar_t *ii, kiss_fft_scalar_t *ro, kiss_fft_scalar_t *io) {
    int n(un);
    vamp_kiss_fft_cfg c = vamp_kiss_fft_alloc(n, false, 0, 0);
    vamp_kiss_fft_cpx *in = new vamp_kiss_fft_cpx[n];
    vamp_kiss_fft_cpx *out = new vamp_kiss_fft_cpx[n];
    for (int i = 0; i < n; ++i) {
        in[i].r = ri[i];
        in[i].i = 0;
    }
    if (ii) {
        for (int i = 0; i < n; ++i) {
            in[i].i = ii[i];
        }
    }
    vamp_kiss_fft(c, in, out);
    for (int i = 0; i < n; ++i) {
        ro[i] = out[i].r;
        io[i] = out[i].i;
    }
    vamp_kiss_fft_free(c);
    delete[] in;
    delete[] out;
}

void fft_inverse(unsigned int un, const kiss_fft_scalar_t *ri, const kiss_fft_scalar_t *ii, kiss_fft_scalar_t *ro, kiss_fft_scalar_t *io) {
    int n(un);
    vamp_kiss_fft_cfg c = vamp_kiss_fft_alloc(n, true, 0, 0);
    vamp_kiss_fft_cpx *in = new vamp_kiss_fft_cpx[n];
    vamp_kiss_fft_cpx *out = new vamp_kiss_fft_cpx[n];
    for (int i = 0; i < n; ++i) {
        in[i].r = ri[i];
        in[i].i = 0;
    }
    if (ii) {
        for (int i = 0; i < n; ++i) {
            in[i].i = ii[i];
        }
    }
    vamp_kiss_fft(c, in, out);
    kiss_fft_scalar_t scale = 1.0 / double(n);
    for (int i = 0; i < n; ++i) {
        ro[i] = out[i].r * scale;
        io[i] = out[i].i * scale;
    }
    vamp_kiss_fft_free(c);
    delete[] in;
    delete[] out;
}

void rfft_forward(unsigned int n, const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co) {
    auto m_fconf = vamp_kiss_fftr_alloc(n, false, 0, 0);
    vamp_kiss_fftr(m_fconf, ri, (vamp_kiss_fft_cpx *)co);
    free(m_fconf);
}

void rfft_inverse(unsigned int n, const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro) {
    auto m_iconf = vamp_kiss_fftr_alloc(n, true, 0, 0);
    vamp_kiss_fftri(m_iconf, (vamp_kiss_fft_cpx *)ci, ro);
    kiss_fft_scalar_t scale = 1.0 / n;
    for (int i = 0; i < n; ++i) {
        ro[i] = ro[i] * scale;
    }
    free(m_iconf);
}

class FFTComplex::D {
public:
    D(int n) :
        m_n(n),
        m_fconf(vamp_kiss_fft_alloc(n, false, 0, 0)),
        m_iconf(vamp_kiss_fft_alloc(n, true, 0, 0)),
        m_ci(new vamp_kiss_fft_cpx[m_n]),
        m_co(new vamp_kiss_fft_cpx[m_n]) { }

    ~D() {
        vamp_kiss_fftr_free(m_fconf);
        vamp_kiss_fftr_free(m_iconf);
        delete[] m_ci;
        delete[] m_co;
    }

    void forward(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co) {
        for (int i = 0; i < m_n; ++i) {
            m_ci[i].r = ci[i*2];
            m_ci[i].i = ci[i*2+1];
        }
        vamp_kiss_fft(m_fconf, m_ci, m_co);
        for (int i = 0; i < m_n; ++i) {
            co[i*2] = m_co[i].r;
            co[i*2+1] = m_co[i].i;
        }
    }

    void inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co) {
        for (int i = 0; i < m_n; ++i) {
            m_ci[i].r = ci[i*2];
            m_ci[i].i = ci[i*2+1];
        }
        vamp_kiss_fft(m_iconf, m_ci, m_co);
        kiss_fft_scalar_t scale = 1.0 / double(m_n);
        for (int i = 0; i < m_n; ++i) {
            co[i*2] = m_co[i].r * scale;
            co[i*2+1] = m_co[i].i * scale;
        }
    }
    
private:
    int m_n;
    vamp_kiss_fft_cfg m_fconf;
    vamp_kiss_fft_cfg m_iconf;
    vamp_kiss_fft_cpx *m_ci;
    vamp_kiss_fft_cpx *m_co;
};

FFTComplex::FFTComplex(unsigned int n) : m_d(new D(n)) { }

FFTComplex::~FFTComplex() {
    delete m_d;
}

void FFTComplex::forward(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co) {
    m_d->forward(ci, co);
}

void FFTComplex::inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co) {
    m_d->inverse(ci, co);
}

class FFTReal::D {
public:
    D(int n) :
        m_n(n),
        m_fconf(vamp_kiss_fftr_alloc(n, false, 0, 0)),
        m_iconf(vamp_kiss_fftr_alloc(n, true, 0, 0)),
        m_ri(new vamp_kiss_fft_scalar[m_n]),
        m_ro(new vamp_kiss_fft_scalar[m_n]),
        m_freq(new vamp_kiss_fft_cpx[n/2+1]) { }

    ~D() {
        vamp_kiss_fftr_free(m_fconf);
        vamp_kiss_fftr_free(m_iconf);
        delete[] m_ri;
        delete[] m_ro;
        delete[] m_freq;
    }

    void forward(const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co) {
        for (int i = 0; i < m_n; ++i) {
            // in case vamp_kiss_fft_scalar is float
            m_ri[i] = ri[i];
        }
        vamp_kiss_fftr(m_fconf, m_ri, m_freq);
        int hs = m_n/2 + 1;
        for (int i = 0; i < hs; ++i) {
            co[i*2] = m_freq[i].r;
            co[i*2+1] = m_freq[i].i;
        }
    }

    void inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro) {
        int hs = m_n/2 + 1;
        for (int i = 0; i < hs; ++i) {
            m_freq[i].r = ci[i*2];
            m_freq[i].i = ci[i*2+1];
        }
        vamp_kiss_fftri(m_iconf, m_freq, m_ro);
        kiss_fft_scalar_t scale = 1.0 / double(m_n);
        for (int i = 0; i < m_n; ++i) {
            ro[i] = m_ro[i] * scale;
        }
    }
    
private:
    int m_n;
    vamp_kiss_fftr_cfg m_fconf;
    vamp_kiss_fftr_cfg m_iconf;
    vamp_kiss_fft_scalar *m_ri;
    vamp_kiss_fft_scalar *m_ro;
    vamp_kiss_fft_cpx *m_freq;
};

FFTReal::FFTReal(unsigned int n) : m_d(new D(n)) { }

FFTReal::~FFTReal() {
    delete m_d;
}

void FFTReal::forward(const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co) {
    m_d->forward(ri, co);
}

void FFTReal::inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro) {
    m_d->inverse(ci, ro);
}

} // namespace vkfft
