#pragma once

#include <vector>

#include "header"

#include "oracle.h"
#include "goldilocks_quadratic_ext.h"
#include "array_view.h"

template <typename field>
class open_param : public oracle<field>{
public:
    open_param(const array_view<field>& mle, const oracle<field> *pcs);
    open_param(const std::vector<int>& log_shape, const oracle<field> *pcs);
    open_param(int num_vars, const oracle<field> *pcs);
    open_param(const open_param<field>& other);

    open_param fix(const std::vector<field>& r) const;

    open_param parse_all(const std::vector<field>& r) const;

    open_param operator()(const std::vector<field>& r) const;
    open_param operator()(const field& r) const;
    open_param operator()(size_t r) const;

    field open(size_t sec_param) const;

    inline field open(const std::vector<field>& z, const size_t& sec_param) const override {
        return parse_all(z).open(sec_param);
    }

    inline int get_num_vars() const override {
        int ret = 0;
        for (int i = next; i < static_cast<int>(shape.size()); ++i) {
            ret += shape[i];
        }
        return ret;
    }
    
    const oracle *pcs;
    std::vector<field> cha; // combined challenges
    std::vector<int> ele; // pointers to the starting element
    std::vector<int> shape;
    std::vector<int> ord;
    std::vector<bool> rev;
    int next;
};

template <typename field>
class equation {
public:
    bool check() const;

    void add(const field& c, const open_param<field>& p);

    std::vector<field> coe;
    std::vector<open_param<field>> par;
    field claim;
};

template <typename field>
class requirement {
public:
    void push_back(const equation<field>& eq);

    bool check() const;

    std::vector<equation<field>> eqs;
};