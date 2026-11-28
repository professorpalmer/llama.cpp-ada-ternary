// Recurrent-state rollback with several sequences whose cells are not in sequence order.
//
// The rollback snapshots of a sequence (n_rs_seq planes) live in the same row as its state. When a micro-batch holds
// sequences whose cells are not next to each other in order, llama_memory_recurrent::find_slot moves cells. A sequence
// that is not in that micro-batch, but lies between them, was moved with its live state only: its snapshots stayed in
// the old row. If that sequence had already been decoded earlier in the same llama_decode call and then rolled back
// (rejected draft tokens), it read another sequence's old state. (NetSphere bench report on Kintsugi-27B, 2026-11-27:
// corrupted answers with a drafter and several requests on one server.)
//
// The test runs the same schedule in two contexts that differ only in the order in which the sequences start:
//   A: 0, 1, 2 (cells in sequence order, no move)      B: 1, 0, 2 (the micro-batch of 1 and 2 moves the cell of 0)
// Schedule: a prefill for each sequence; ONE decode call with sequence 0 one token longer than sequences 1 and 2 (so
// sequence 0 runs first, alone); a rollback of sequence 0 by r tokens; then more tokens of sequence 0. The logits of
// sequence 0 must be the same in A and B for every r in 0 .. n_rs_seq. A case with two sequences (no cell between
// them can exist) runs too.
//
//   test-recurrent-cell-move -m model.gguf [-ngl 99]        env TEST_N_RS_SEQ (default 2), TEST_EPS (default 1e-4)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static llama_token tok(uint32_t seq, llama_pos pos, int n_vocab) {
    return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
}

// returns the logits of the continuation rows of sequence 0 (n_cont rows of n_vocab), empty on failure
static std::vector<float> run_schedule(const common_params & params, llama_model * model, int n_vocab,
        const std::vector<llama_seq_id> & start_order, uint32_t n_rs_seq, uint32_t n_reject) {
    const uint32_t n_seqs   = start_order.size();
    const uint32_t n_prompt = n_rs_seq + 5;
    const uint32_t n_short  = n_rs_seq + 1; // sequences 1..: a full set of snapshots
    const uint32_t n_long   = n_rs_seq + 2; // sequence 0: one more, so the split runs it first and alone
    const uint32_t n_cont   = 4;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seqs;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_ctx      = 128*n_seqs;
    cparams.n_batch    = 128;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = false;

    std::vector<float> out;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init the context\n", __func__);
        return out;
    }
    if (llama_n_rs_seq(ctx) < n_rs_seq) {
        fprintf(stderr, "%s : n_rs_seq %u is not available (%u)\n", __func__, n_rs_seq, llama_n_rs_seq(ctx));
        llama_free(ctx);
        return out;
    }

    bool ok = true;
    llama_batch batch = llama_batch_init(128, 0, 1);

    // prefill, one decode call for each sequence, in the start order (this fixes the cell of each sequence)
    for (llama_seq_id s : start_order) {
        common_batch_clear(batch);
        for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos, n_vocab), pos, { s }, false);
        }
        ok = ok && llama_decode(ctx, batch) == 0;
    }

    // one decode call: sequence 0 (longer) first, then the other sequences
    common_batch_clear(batch);
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seqs; ++s) {
        const uint32_t n = s == 0 ? n_long : n_short;
        for (uint32_t i = 0; i < n; ++i) {
            const llama_pos pos = n_prompt + i;
            common_batch_add(batch, tok(s, pos, n_vocab), pos, { s }, false);
        }
    }
    ok = ok && llama_decode(ctx, batch) == 0;

    // sequence 0 rejects its last n_reject tokens
    const llama_pos p0 = (llama_pos) (n_prompt + n_long - n_reject);
    if (ok && n_reject > 0) {
        ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1);
        if (!ok) {
            fprintf(stderr, "%s : the rollback of %u tokens was refused\n", __func__, n_reject);
        }
    }

    // sequence 0 continues
    common_batch_clear(batch);
    for (uint32_t i = 0; i < n_cont; ++i) {
        const llama_pos pos = p0 + (llama_pos) i;
        common_batch_add(batch, tok(0, pos + 3, n_vocab), pos, { 0 }, true);
    }
    ok = ok && llama_decode(ctx, batch) == 0;

    if (ok) {
        for (uint32_t i = 0; i < n_cont && ok; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            ok = l != nullptr;
            if (ok) {
                out.insert(out.end(), l, l + n_vocab);
            }
        }
    }
    if (!ok) {
        out.clear();
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load the model\n", __func__);
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        llama_model_free(model);
        return 0;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const char * e_rs  = getenv("TEST_N_RS_SEQ");
    const char * e_eps = getenv("TEST_EPS");
    const uint32_t n_rs_seq = e_rs  ? (uint32_t) atoi(e_rs) : 2;
    const float    eps      = e_eps ? (float) atof(e_eps)   : 1e-4f;

    struct test_case {
        const char * name;
        std::vector<llama_seq_id> order_a;
        std::vector<llama_seq_id> order_b;
    };
    const std::vector<test_case> cases = {
        { "3 sequences, start order 0,1,2 against 1,0,2", { 0, 1, 2 }, { 1, 0, 2 } },
        { "2 sequences, start order 0,1 against 1,0",     { 0, 1 },    { 1, 0 }    },
    };

    int n_fail = 0;
    for (const auto & c : cases) {
        for (uint32_t r = 0; r <= n_rs_seq; ++r) {
            const std::vector<float> la = run_schedule(params, model, n_vocab, c.order_a, n_rs_seq, r);
            const std::vector<float> lb = run_schedule(params, model, n_vocab, c.order_b, n_rs_seq, r);
            if (la.empty() || lb.empty() || la.size() != lb.size()) {
                fprintf(stderr, "FAIL  %s, rollback %u: the schedule did not run\n", c.name, r);
                n_fail++;
                continue;
            }
            float  diff_max = 0.0f;
            size_t n_bad    = 0;
            for (size_t i = 0; i < la.size(); ++i) {
                const float d = std::fabs(la[i] - lb[i]);
                if (!(d <= eps)) {
                    n_bad++;
                }
                if (d > diff_max || std::isnan(d)) {
                    diff_max = d;
                }
            }
            const bool pass = n_bad == 0;
            fprintf(stderr, "%s  %s, rollback %u of %u: max logit difference %g (%zu of %zu values over %g)\n",
                    pass ? "pass" : "FAIL", c.name, r, n_rs_seq, (double) diff_max, n_bad, la.size(), (double) eps);
            n_fail += pass ? 0 : 1;
        }
    }

    llama_model_free(model);
    fprintf(stderr, "%s : %s (%d failed)\n", __func__, n_fail == 0 ? "all cases passed" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
