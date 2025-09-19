#include "header"
#include "mle_open.h"
#include "util.h"

template <typename field>
open_param<field>::open_param(const array_view<field>& mle, const oracle *pcs)
    : next(0), pcs(pcs)
{
    int n = int(mle.get_shape().size()), sz = 0;
    shape.resize(n);
    ord = mle.get_order();
    rev = mle.get_reversed();
    std::vector<int> start(n + 1);
    for (size_t i = 0; i < n; ++i) {
        shape[i] = find_ceiling_log2(mle.shape(i));
        sz += shape[i];
        start[ord[i] + 1] = shape[i];
    }
    for (int i = 1; i < n; ++i) {
        start[i] += start[i - 1];
    }
    cha.resize(sz);
    ele.resize(n);
    for (size_t i = 0; i < n; ++i) {
        ele[i] = start[ord[i]];
    }
}

template <typename field>
open_param<field>::open_param(const std::vector<int>& log_shape, const oracle<field>* pcs)
    : next(0), pcs(pcs), shape(log_shape) {
    int n = int(shape.size());
    ord.resize(n);
    rev.resize(n);
    for (int i = 0; i < n; ++i) {
        ord[i] = i;
        rev[i] = false;
    }
    int sz = 0;
    std::vector<int> start(n + 1);
    for (size_t i = 0; i < n; ++i) {
        sz += shape[i];
        start[ord[i] + 1] = shape[i];
    }
    for (int i = 1; i < n; ++i) {
        start[i] += start[i - 1];
    }
    cha.resize(sz);
    ele.resize(n);
    for (size_t i = 0; i < n; ++i) {
        ele[i] = start[ord[i]];
    }
}

template <typename field>
open_param<field>::open_param(int num_vars, const oracle<field>* pcs)
    : open_param(std::vector<int>{num_vars}, pcs)
{
}

template <typename field>
open_param<field>::open_param(const open_param<field>& other)
    : pcs(other.pcs), cha(other.cha), ele(other.ele), shape(other.shape), ord(other.ord), rev(other.rev), next(other.next)
{
}

template <typename field>
open_param<field> open_param<field>::fix(const std::vector<field>& r) const {
    assert(next != int(shape.size()));
    assert(r.size() == shape[next]);

    open_param res(*this);
    for (int i = 0; i < shape[next]; ++i) {
        res.cha[res.ele[next] + i] = rev[next] ? Goldilocks2::one() - r[i] : r[i];
    }
    ++res.next;
    return res;
}

template <typename field>
open_param<field> open_param<field>::parse_all(const std::vector<field>& r) const {
    open_param res(*this);
    int k = 0;
    for (int i = next; i < int(shape.size()); ++i) {
        for (int j = 0; j < shape[i]; ++j) {
            res.cha[res.ele[i] + j] = rev[i] ? Goldilocks2::one() - r[k] : r[k];
            ++k;
        }
    }
    assert(k == int(r.size()));
    res.next = int(shape.size());
    return res;
}

template <typename field>
open_param<field> open_param<field>::operator()(const std::vector<field>& r) const {
    return fix(r);
}

template <typename field>
open_param<field> open_param<field>::operator()(const field& r) const {
    return fix({r});
}

template <typename field>
open_param<field> open_param<field>::operator()(size_t r) const {
    int len = shape[next];
    std::vector<field> vec(len);
    for (int i = 0; i < len; ++i) {
        vec[i] = ((r >> (len - i - 1)) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
    }
    return fix(vec);
}

template <typename field>
field open_param<field>::open(size_t sec_param) const {
    return pcs->open(cha, sec_param);
}

template <typename field>
bool equation<field>::check() const {
    int n = int(coe.size());
    auto sum = Goldilocks2::zero();
    for (int i = 0; i != n; ++i) {
        sum += coe[i] * par[i].pcs->open(par[i].cha, 0);
    }
    return sum == claim;
}

template <typename field>
void equation<field>::add(const field& c, const open_param<field>& p) {
    coe.push_back(c);
    par.push_back(p);
}

template <typename field>
void requirement<field>::push_back(const equation<field>& eq) {
    eqs.push_back(eq);
}

template <typename field>
bool requirement<field>::check() const {
    for (const auto& eq : eqs) {
        if (!eq.check()) {
            return false;
        }
    }
    return true;
}