#pragma once

#include <vector>

#include "header"

#include "oracle.h"
#include "goldilocks_quadratic_ext.h"
#include "array_view.h"

class open_param{
public:
    open_param(const array_view<Goldilocks2::Element>& mle, const oracle *pcs);
    open_param(const open_param& other);

    open_param fix(const std::vector<Goldilocks2::Element>& r) const;

    open_param parse_all(const std::vector<Goldilocks2::Element>& r) const;

    open_param operator()(const std::vector<Goldilocks2::Element>& r) const;

    Goldilocks2::Element open(size_t sec_param) const;

    const oracle *pcs;
    std::vector<Goldilocks2::Element> cha; // combined challenges
    std::vector<int> ele; // pointers to the starting element
    std::vector<int> shape;
    std::vector<int> ord;
    std::vector<bool> rev;
    int next;
};


class equation {
public:
    bool check() const;

    void add(const Goldilocks2::Element& c, const open_param& p); 

    std::vector<Goldilocks2::Element> coe;
    std::vector<open_param> par;
    Goldilocks2::Element claim;
};

class requirement {
public:
    void push_back(const equation& eq);

    bool check() const;

    std::vector<equation> eqs;
};