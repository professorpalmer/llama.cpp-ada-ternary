#include "common.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// GGML_CUDA_KV_TIER_KFIRST_STATS=1: stats-only probe for a "K first" decode path over a tiered KV cache. For an
// attention op with at most 8 queries whose K range reaches the host part of a tiered buffer, it computes the scaled
// scores of every cell (VRAM and host), the softmax per (token, q head), and for each eps the number of host cells
// that a second pass would have to read V for so that the dropped host mass is <= eps (per q head, and as the union
// over the q heads and tokens of each KV head). It runs before the normal attention, reads K in place, writes only
// its own scratch memory and synchronizes the stream; the attention result is unchanged. CUDA graphs are off in
// this mode (common.cuh). GGML_CUDA_KV_TIER_KFIRST_EVERY=K: one log line per K probed ops (default 64), and a
// per-cache summary of the last 16*K ops.

// ggml-cuda.cu
size_t ggml_cuda_tier_first_host(const void * ptr, size_t nbytes);

#define KFIRST_NB      256 // buckets of 1/4 in log2(w): bucket b holds w in (2^-(b+1)/4, 2^-b/4]; the last one all below
#define KFIRST_NEPS    4
#define KFIRST_MAX_D   512
#define KFIRST_MAX_KVH 16
#define KFIRST_MAX_Q   8

static const char * kfirst_eps_name[KFIRST_NEPS] = { "1e-2", "1e-3", "1e-4", "1e-5" };

static __device__ __forceinline__ float kfirst_eps(const int e) {
    return e == 0 ? 1e-2f : e == 1 ? 1e-3f : e == 2 ? 1e-4f : 1e-5f;
}

// bucket of w = exp(s - lse), s finite
static __device__ __forceinline__ int kfirst_bucket(const float s, const float lse) {
    const float x = (lse - s) * (4.0f*1.4426950409f);
    return x < (float) KFIRST_NB ? max(0, (int) x) : KFIRST_NB - 1;
}

struct kfirst_args {
    const char  * Q;
    size_t        q_nb1, q_nb2;
    const char  * K;
    size_t        k_nb1, k_nb2;
    const char  * mask;
    size_t        m_nb1, m_nb2;
    int           m_ne2;
    bool          m_packed;
    float         scale, softcap, max_bias, m0, m1;
    uint32_t      n_head_log2;
    int           D, n_tok, H, n_kvh, G;
    int64_t       n_kv;
};

template <ggml_type type>
static __device__ __forceinline__ float kfirst_k_value(const char * row, const int e) {
    if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) row + e/QK4_0;
        const int i = e % QK4_0;
        const int q = i < QK4_0/2 ? (b->qs[i] & 0x0F) : (b->qs[i - QK4_0/2] >> 4);
        return (q - 8) * __half2float(b->d);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row + e/QK8_0;
        return b->qs[e % QK8_0] * __half2float(b->d);
    } else {
        return __half2float(((const half *) row)[e]);
    }
}

// one warp per (cell, KV head): dequantize the K row once, then the score of each (token, q head) of its group.
// S is [n_tok*H, n_kv], row j*H + h.
template <ggml_type type>
static __global__ void k_kfirst_scores(const kfirst_args a, float * __restrict__ S) {
    const int64_t w    = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x) / WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;
    if (w >= a.n_kv*a.n_kvh) {
        return;
    }
    const int64_t c = w / a.n_kvh;
    const int     g = w % a.n_kvh;
    const char * row = a.K + c*a.k_nb1 + g*a.k_nb2;

    float k[KFIRST_MAX_D/WARP_SIZE];
#pragma unroll
    for (int t = 0; t < KFIRST_MAX_D/WARP_SIZE; ++t) {
        k[t] = t*WARP_SIZE < a.D ? kfirst_k_value<type>(row, t*WARP_SIZE + lane) : 0.0f;
    }

    for (int j = 0; j < a.n_tok; ++j) {
        for (int hh = 0; hh < a.G; ++hh) {
            const int h = g*a.G + hh;
            const float * q = (const float *) (a.Q + j*a.q_nb1 + h*a.q_nb2);
            float sum = 0.0f;
#pragma unroll
            for (int t = 0; t < KFIRST_MAX_D/WARP_SIZE; ++t) {
                if (t*WARP_SIZE < a.D) {
                    sum += k[t]*q[t*WARP_SIZE + lane];
                }
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                float s = sum*a.scale;
                if (a.softcap != 0.0f) {
                    s = a.softcap*tanhf(s);
                }
                if (a.mask) {
                    const char * mrow = a.mask + j*a.m_nb1 + (h % a.m_ne2)*a.m_nb2;
                    const float mv = a.m_packed ?
                        ((((const uint32_t *) mrow)[c/32] >> (c%32)) & 1u ? 0.0f : -INFINITY) :
                        __half2float(((const half *) mrow)[c]);
                    s += get_alibi_slope(a.max_bias, h, a.n_head_log2, a.m0, a.m1)*mv;
                }
                S[((int64_t) j*a.H + h)*a.n_kv + c] = s;
            }
        }
    }
}

// result valid in every thread; buf: 32 floats of shared memory
template <bool is_max>
static __device__ float kfirst_block_reduce(float x, float * buf) {
    x = is_max ? warp_reduce_max(x) : warp_reduce_sum(x);
    const int lane = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (lane == 0) {
        buf[threadIdx.x / WARP_SIZE] = x;
    }
    __syncthreads();
    x = lane < (int) (blockDim.x / WARP_SIZE) ? buf[lane] : (is_max ? -INFINITY : 0.0f);
    return is_max ? warp_reduce_max(x) : warp_reduce_sum(x);
}

// one block per row: log of the softmax sum over all cells (plus the sink), host mass, finite host cells
static __global__ void k_kfirst_softmax(const float * __restrict__ S, const float * __restrict__ sinks, const int H,
        const int64_t n_kv, const int64_t h0, float * __restrict__ row_lse, float * __restrict__ host_mass, uint32_t * __restrict__ row_valid) {
    __shared__ float buf[32];
    const int r = blockIdx.x;
    const float * s = S + (int64_t) r*n_kv;

    float m = -INFINITY;
    for (int64_t c = threadIdx.x; c < n_kv; c += blockDim.x) {
        m = fmaxf(m, s[c]);
    }
    m = kfirst_block_reduce<true>(m, buf);
    if (sinks) {
        m = fmaxf(m, sinks[r % H]);
    }

    float sum = 0.0f, hsum = 0.0f, nv = 0.0f;
    if (m != -INFINITY) {
        for (int64_t c = threadIdx.x; c < n_kv; c += blockDim.x) {
            const float e = expf(s[c] - m);
            sum += e;
            if (c >= h0) {
                hsum += e;
                nv   += s[c] != -INFINITY;
            }
        }
    }
    sum  = kfirst_block_reduce<false>(sum,  buf);
    hsum = kfirst_block_reduce<false>(hsum, buf);
    nv   = kfirst_block_reduce<false>(nv,   buf);
    if (threadIdx.x == 0) {
        if (sinks && m != -INFINITY) {
            sum += expf(sinks[r % H] - m);
        }
        row_lse[r]   = sum > 0.0f ? m + logf(sum) : -INFINITY;
        host_mass[r] = sum > 0.0f ? hsum/sum : 0.0f;
        row_valid[r] = (uint32_t) nv;
    }
}

// grid (chunks, rows): mass and count of the host cells per log-weight bucket
static __global__ void k_kfirst_hist(const float * __restrict__ S, const float * __restrict__ row_lse, const int64_t n_kv,
        const int64_t h0, float * __restrict__ hmass, uint32_t * __restrict__ hcnt) {
    __shared__ float    sm[KFIRST_NB];
    __shared__ uint32_t sc[KFIRST_NB];
    for (int i = threadIdx.x; i < KFIRST_NB; i += blockDim.x) {
        sm[i] = 0.0f;
        sc[i] = 0;
    }
    __syncthreads();
    const int r = blockIdx.y;
    const float lse = row_lse[r];
    const float * s = S + (int64_t) r*n_kv;
    if (lse != -INFINITY) {
        for (int64_t c = h0 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; c < n_kv; c += (int64_t) gridDim.x*blockDim.x) {
            const float v = s[c];
            if (v == -INFINITY) {
                continue;
            }
            const int b = kfirst_bucket(v, lse);
            atomicAdd(&sm[b], expf(v - lse));
            atomicAdd(&sc[b], 1u);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < KFIRST_NB; i += blockDim.x) {
        if (sc[i]) {
            atomicAdd(&hmass[r*KFIRST_NB + i], sm[i]);
            atomicAdd(&hcnt [r*KFIRST_NB + i], sc[i]);
        }
    }
}

// one thread per (row, eps): drop the lowest buckets while their total mass stays <= eps; bkeep = last kept bucket
// (-1: drop every host cell), kept = host cells in the kept buckets
static __global__ void k_kfirst_thresh(const float * __restrict__ hmass, const uint32_t * __restrict__ hcnt, const int n_rows,
        int32_t * __restrict__ bkeep, uint32_t * __restrict__ kept) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows*KFIRST_NEPS) {
        return;
    }
    const int   r   = i / KFIRST_NEPS;
    const float eps = kfirst_eps(i % KFIRST_NEPS);
    float cum = 0.0f;
    int   B   = -1;
    for (int b = KFIRST_NB - 1; b >= 0; --b) {
        cum += hmass[r*KFIRST_NB + b];
        if (cum > eps) {
            B = b;
            break;
        }
    }
    uint32_t k = 0;
    for (int b = 0; b <= B; ++b) {
        k += hcnt[r*KFIRST_NB + b];
    }
    bkeep[i] = B;
    kept[i]  = k;
}

// one thread per host cell: per KV head, the cell is kept for eps when any of its (token, q head) rows keeps it.
// out[g*(NEPS + 1) + e]: kept cells, out[g*(NEPS + 1) + NEPS]: cells with a finite score in any of the rows
static __global__ void k_kfirst_union(const float * __restrict__ S, const float * __restrict__ row_lse, const int32_t * __restrict__ bkeep,
        const int n_tok, const int H, const int n_kvh, const int G, const int64_t n_kv, const int64_t h0, uint32_t * __restrict__ out) {
    __shared__ uint32_t su[KFIRST_MAX_KVH*(KFIRST_NEPS + 1)];
    for (int i = threadIdx.x; i < n_kvh*(KFIRST_NEPS + 1); i += blockDim.x) {
        su[i] = 0;
    }
    __syncthreads();
    const int64_t c = h0 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n_kv) {
        for (int g = 0; g < n_kvh; ++g) {
            bool valid = false;
            bool keep[KFIRST_NEPS] = { false };
            for (int j = 0; j < n_tok; ++j) {
                for (int hh = 0; hh < G; ++hh) {
                    const int   r   = j*H + g*G + hh;
                    const float v   = S[(int64_t) r*n_kv + c];
                    const float lse = row_lse[r];
                    if (v == -INFINITY || lse == -INFINITY) {
                        continue;
                    }
                    valid = true;
                    const int b = kfirst_bucket(v, lse);
#pragma unroll
                    for (int e = 0; e < KFIRST_NEPS; ++e) {
                        keep[e] = keep[e] || b <= bkeep[r*KFIRST_NEPS + e];
                    }
                }
            }
            if (valid) {
                atomicAdd(&su[g*(KFIRST_NEPS + 1) + KFIRST_NEPS], 1u);
            }
#pragma unroll
            for (int e = 0; e < KFIRST_NEPS; ++e) {
                if (keep[e]) {
                    atomicAdd(&su[g*(KFIRST_NEPS + 1) + e], 1u);
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_kvh*(KFIRST_NEPS + 1); i += blockDim.x) {
        if (su[i]) {
            atomicAdd(&out[i], su[i]);
        }
    }
}

bool ggml_cuda_kfirst_stats_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_KV_TIER_KFIRST_STATS");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static int64_t ggml_cuda_kfirst_every() {
    static const int64_t k = [] {
        const char * e = getenv("GGML_CUDA_KV_TIER_KFIRST_EVERY");
        const int64_t v = e ? (int64_t) atoll(e) : 64;
        return v > 0 ? v : (int64_t) 64;
    }();
    return k;
}

// per-cache sums over the ops of one summary window
struct kfirst_acc {
    std::string name;
    int64_t n_ops = 0;
    double  n_kv = 0.0, host = 0.0, mass = 0.0;
    double  u_mean[KFIRST_NEPS] = {}, u_max[KFIRST_NEPS] = {}, h_mean[KFIRST_NEPS] = {};
    double  u_head[8][KFIRST_NEPS] = {};   // per KV head (calibration of streaming vs retrieval heads)
    int     n_kvh = 0;
};

static void kfirst_warn_once(const char * what) {
    static std::mutex m;
    static std::map<std::string, bool> seen;
    std::lock_guard<std::mutex> lock(m);
    if (!seen[what]) {
        seen[what] = true;
        GGML_LOG_WARN("%s: GGML_CUDA_KV_TIER_KFIRST_STATS: op skipped: %s\n", __func__, what);
    }
}

void ggml_cuda_kfirst_probe(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!Q || !K || Q->ne[1] > KFIRST_MAX_Q) {
        return;
    }
    const int64_t n_kv = K->ne[1];
    const size_t  off  = ggml_cuda_tier_first_host(K->data, ggml_nbytes(K));
    if (off >= ggml_nbytes(K)) {
        return; // not tiered, or all cells in VRAM
    }
    // cells are the outer dimension of each K row group: [D, n_kv, n_kvh] with nb2*n_kvh <= nb1
    if (K->nb[1] < K->nb[2]*K->ne[2]) {
        kfirst_warn_once("K layout is not cell-major");
        return;
    }
    const int64_t h0 = (int64_t) (off / K->nb[1]);
    if (h0 >= n_kv) {
        return;
    }

    const int D     = (int) K->ne[0];
    const int n_tok = (int) Q->ne[1];
    const int H     = (int) Q->ne[2];
    const int n_kvh = (int) K->ne[2];
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->ne[0] != D) {
        kfirst_warn_once("Q is not f32 with K's head size");
        return;
    }
    if (D % WARP_SIZE != 0 || D > KFIRST_MAX_D || n_kvh > KFIRST_MAX_KVH || H % n_kvh != 0) {
        kfirst_warn_once("head size or head counts not supported");
        return;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || (mask && mask->ne[3] != 1)) {
        kfirst_warn_once("more than one sequence");
        return;
    }
    if (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_F16) {
        kfirst_warn_once("K type not supported (q4_0, q8_0, f16)");
        return;
    }
    if (K->nb[0] != ggml_type_size(K->type)) {
        kfirst_warn_once("K rows not contiguous");
        return;
    }
    if (mask && !((mask->type == GGML_TYPE_F16 && mask->ne[0] >= n_kv) || (mask->type == GGML_TYPE_I32 && mask->ne[0]*32 >= n_kv)) ) {
        kfirst_warn_once("mask type or size not supported");
        return;
    }
    if (sinks && sinks->type != GGML_TYPE_F32) {
        kfirst_warn_once("sinks are not f32");
        return;
    }

    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        kfirst_warn_once("stream is being captured into a CUDA graph");
        return;
    }

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // scratch: S, then the 32-bit words [hmass | hcnt | uni] (zeroed), [host_mass | row_valid | kept] (copied to the
    // host together with uni), [row_lse | bkeep]
    const int64_t rows    = (int64_t) n_tok*H;
    const size_t  s_bytes = GGML_PAD(rows*n_kv*sizeof(float), 256);
    const size_t  n_hist  = rows*KFIRST_NB;
    const size_t  n_uni   = (size_t) n_kvh*(KFIRST_NEPS + 1);
    const size_t  n_copy  = n_uni + rows + rows + rows*KFIRST_NEPS;
    const size_t  n_words = 2*n_hist + n_copy + rows + rows*KFIRST_NEPS;
    const size_t  need    = s_bytes + n_words*4;

    static char * buf[GGML_CUDA_MAX_DEVICES] = { nullptr };
    static size_t buf_size[GGML_CUDA_MAX_DEVICES] = { 0 };
    const int dev = ctx.device;
    if (buf_size[dev] < need) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (buf[dev]) {
            CUDA_CHECK(cudaFree(buf[dev]));
            buf[dev] = nullptr;
            buf_size[dev] = 0;
        }
        const size_t want = need + need/4;
        if (cudaMalloc(&buf[dev], want) != cudaSuccess) {
            (void) cudaGetLastError();
            buf[dev] = nullptr;
            GGML_LOG_WARN("%s: GGML_CUDA_KV_TIER_KFIRST_STATS: no VRAM for %.1f MiB of scratch, op skipped\n", __func__, want/1048576.0);
            return;
        }
        buf_size[dev] = want;
    }

    float    * S         = (float *) buf[dev];
    uint32_t * words     = (uint32_t *) (buf[dev] + s_bytes);
    float    * hmass     = (float *) words;
    uint32_t * hcnt      = words + n_hist;
    uint32_t * uni       = hcnt + n_hist;
    float    * host_mass = (float *) (uni + n_uni);
    uint32_t * row_valid = (uint32_t *) (host_mass + rows);
    uint32_t * kept      = row_valid + rows;
    float    * row_lse   = (float *) (kept + rows*KFIRST_NEPS);
    int32_t  * bkeep     = (int32_t *) (row_lse + rows);

    kfirst_args a;
    a.Q        = (const char *) Q->data;
    a.q_nb1    = Q->nb[1];
    a.q_nb2    = Q->nb[2];
    a.K        = (const char *) K->data;
    a.k_nb1    = K->nb[1];
    a.k_nb2    = K->nb[2];
    a.mask     = mask ? (const char *) mask->data : nullptr;
    a.m_nb1    = mask ? mask->nb[1] : 0;
    a.m_nb2    = mask ? mask->nb[2] : 0;
    a.m_ne2    = mask ? (int) mask->ne[2] : 1;
    a.m_packed = mask && mask->type == GGML_TYPE_I32;
    memcpy(&a.scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&a.max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&a.softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (a.softcap != 0.0f) {
        a.scale /= a.softcap;
    }
    a.n_head_log2 = 1u << uint32_t(floorf(log2f(float(H))));
    a.m0       = powf(2.0f, -(a.max_bias       ) / a.n_head_log2);
    a.m1       = powf(2.0f, -(a.max_bias / 2.0f) / a.n_head_log2);
    a.D        = D;
    a.n_tok    = n_tok;
    a.H        = H;
    a.n_kvh    = n_kvh;
    a.G        = H / n_kvh;
    a.n_kv     = n_kv;

    CUDA_CHECK(cudaMemsetAsync(words, 0, (2*n_hist + n_uni)*4, stream));

    const int64_t n_warps = n_kv*n_kvh;
    const int     wpb     = 4;
    const dim3    grid_s((unsigned) ((n_warps + wpb - 1)/wpb));
    switch (K->type) {
        case GGML_TYPE_Q4_0: k_kfirst_scores<GGML_TYPE_Q4_0><<<grid_s, wpb*WARP_SIZE, 0, stream>>>(a, S); break;
        case GGML_TYPE_Q8_0: k_kfirst_scores<GGML_TYPE_Q8_0><<<grid_s, wpb*WARP_SIZE, 0, stream>>>(a, S); break;
        default:             k_kfirst_scores<GGML_TYPE_F16 ><<<grid_s, wpb*WARP_SIZE, 0, stream>>>(a, S); break;
    }
    CUDA_CHECK(cudaGetLastError());

    k_kfirst_softmax<<<(unsigned) rows, 256, 0, stream>>>(S, sinks ? (const float *) sinks->data : nullptr, H, n_kv, h0,
        row_lse, host_mass, row_valid);
    CUDA_CHECK(cudaGetLastError());

    const int64_t n_host = n_kv - h0;
    const dim3 grid_h((unsigned) std::min<int64_t>(32, (n_host + 255)/256), (unsigned) rows);
    k_kfirst_hist<<<grid_h, 256, 0, stream>>>(S, row_lse, n_kv, h0, hmass, hcnt);
    CUDA_CHECK(cudaGetLastError());

    k_kfirst_thresh<<<(unsigned) ((rows*KFIRST_NEPS + 127)/128), 128, 0, stream>>>(hmass, hcnt, (int) rows, bkeep, kept);
    CUDA_CHECK(cudaGetLastError());

    k_kfirst_union<<<(unsigned) ((n_host + 255)/256), 256, 0, stream>>>(S, row_lse, bkeep, n_tok, H, n_kvh, a.G, n_kv, h0, uni);
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint32_t> res(n_copy);
    CUDA_CHECK(cudaMemcpyAsync(res.data(), uni, n_copy*4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const uint32_t * r_uni   = res.data();
    const uint32_t * r_mass  = r_uni + n_uni;
    const uint32_t * r_valid = r_mass + rows;
    const uint32_t * r_kept  = r_valid + rows;

    // per op: host mass and per-head fractions as means over the rows, union fractions as mean and max over KV heads
    double mass = 0.0;
    double h_mean[KFIRST_NEPS] = {};
    double u_mean[KFIRST_NEPS] = {};
    double u_max [KFIRST_NEPS] = {};
    for (int64_t r = 0; r < rows; ++r) {
        float m;
        memcpy(&m, &r_mass[r], sizeof(float));
        mass += m / rows;
        for (int e = 0; e < KFIRST_NEPS; ++e) {
            h_mean[e] += r_valid[r] ? (double) r_kept[r*KFIRST_NEPS + e] / r_valid[r] / rows : 0.0;
        }
    }
    uint32_t valid_max = 0;
    double u_head_op[8][KFIRST_NEPS] = {};
    for (int g = 0; g < n_kvh; ++g) {
        const uint32_t v = r_uni[g*(KFIRST_NEPS + 1) + KFIRST_NEPS];
        valid_max = std::max(valid_max, v);
        for (int e = 0; e < KFIRST_NEPS; ++e) {
            const double f = v ? (double) r_uni[g*(KFIRST_NEPS + 1) + e] / v : 0.0;
            u_mean[e] += f / n_kvh;
            u_max [e]  = std::max(u_max[e], f);
            u_head_op[std::min(g, 7)][e] = f;
        }
    }

    const ggml_tensor * root = K->view_src ? K->view_src : K;
    static int64_t n_ops = 0;
    static std::map<const ggml_tensor *, kfirst_acc> accs;
    kfirst_acc & acc = accs[root];
    acc.name   = root->name;
    acc.n_ops += 1;
    acc.n_kv  += (double) n_kv;
    acc.host  += (double) n_host;
    acc.mass  += mass;
    for (int e = 0; e < KFIRST_NEPS; ++e) {
        acc.u_mean[e] += u_mean[e];
        acc.u_max [e]  = std::max(acc.u_max[e], u_max[e]);
        acc.h_mean[e] += h_mean[e];
        for (int g = 0; g < std::min(n_kvh, 8); ++g) {
            acc.u_head[g][e] += u_head_op[g][e];
        }
    }
    acc.n_kvh = std::min(n_kvh, 8);

    const int64_t every = ggml_cuda_kfirst_every();
    if (n_ops % every == 0) {
        char line[512];
        int  n = snprintf(line, sizeof(line), "kfirst op %lld %s n_kv %lld host %lld valid %u q %d mass %.4f",
            (long long) n_ops, root->name, (long long) n_kv, (long long) n_host, valid_max, n_tok, mass);
        for (int e = 0; e < KFIRST_NEPS && n > 0 && n < (int) sizeof(line); ++e) {
            n += snprintf(line + n, sizeof(line) - n, " | %s u %.4f/%.4f h %.4f", kfirst_eps_name[e], u_mean[e], u_max[e], h_mean[e]);
        }
        GGML_LOG_WARN("%s\n", line);
    }
    n_ops += 1;
    if (n_ops % (16*every) == 0) {
        for (auto & it : accs) {
            const kfirst_acc & s = it.second;
            if (s.n_ops == 0) {
                continue;
            }
            char line[512];
            int  n = snprintf(line, sizeof(line), "kfirst-sum %s ops %lld n_kv %.0f host %.0f mass %.4f", s.name.c_str(),
                (long long) s.n_ops, s.n_kv/s.n_ops, s.host/s.n_ops, s.mass/s.n_ops);
            for (int e = 0; e < KFIRST_NEPS && n > 0 && n < (int) sizeof(line); ++e) {
                n += snprintf(line + n, sizeof(line) - n, " | %s u %.4f/%.4f h %.4f", kfirst_eps_name[e],
                    s.u_mean[e]/s.n_ops, s.u_max[e], s.h_mean[e]/s.n_ops);
            }
            GGML_LOG_WARN("%s\n", line);
            // per KV head union fraction (mean over ops), for the streaming/retrieval head calibration
            char hl[512];
            int  m = snprintf(hl, sizeof(hl), "kfirst-heads %s", s.name.c_str());
            for (int e = 0; e < KFIRST_NEPS && m > 0 && m < (int) sizeof(hl); ++e) {
                m += snprintf(hl + m, sizeof(hl) - m, " | %s", kfirst_eps_name[e]);
                for (int g = 0; g < s.n_kvh && m > 0 && m < (int) sizeof(hl); ++g) {
                    m += snprintf(hl + m, sizeof(hl) - m, " %.4f", s.u_head[g][e]/s.n_ops);
                }
            }
            GGML_LOG_WARN("%s\n", hl);
        }
        accs.clear();
    }
}
