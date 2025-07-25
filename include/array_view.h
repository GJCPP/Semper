#pragma once

#include <cassert>

#include <iostream>
#include <array>
#include <vector>

template<typename T>
class array_view;

template<typename T>
std::ostream& operator<<(std::ostream& os, const array_view<T>& arr);

template<typename T>
class array;

template<typename T>
class array_view {
public:
    friend std::ostream& operator<< <T>(std::ostream& os, const array_view<T>& arr);
    friend class array<T>;

    array_view() {}

    array_view(T* _data, const std::vector<size_t>& _shape, const std::vector<size_t>& _offset = {}, const std::vector<bool>& _reversed = {}, const std::vector<int>& _order = {})
        : dims(_shape.size()), data(_data), data_size(1), data_shape(_shape) {
        for (size_t i = 0; i < _shape.size(); ++i) {
            data_size *= _shape[i];
        }
        if (_offset.size() == 0) {
            init_offset();
        }
        else {
            index_offset = _offset;
        }
        if (_reversed.size() == 0) {
            reversed = std::vector<bool>(_shape.size(), false);
        }
        else {
            reversed = _reversed;
        }
        if (_order.size() == 0) {
            order = std::vector<int>(_shape.size());
            for (size_t i = 0; i < _shape.size(); ++i) {
                order[i] = i;
            }
        }
        else {
            order = _order;
        }
    }

    T& get(size_t index) {
        return data[index];
    }

    const T& get(size_t index) const {
        return data[index];
    }

    void reorder(const std::vector<int>& ind) {
        if (ind.size() != static_cast<size_t>(dims)) {
            throw std::runtime_error("array_view: Indices size does not match");
        }
        std::vector<size_t> new_shape(dims);
        std::vector<size_t> new_offset(dims);
        std::vector<bool> new_reversed(dims);
        for (int i = 0; i < dims; ++i) {
            new_shape[i] = data_shape[ind[i]];
            new_offset[i] = index_offset[ind[i]];
            new_reversed[i] = reversed[ind[i]];
        }
        data_shape = new_shape;
        index_offset = new_offset;
        reversed = new_reversed;
        order = ind;
    }

    void swap_dim(int i, int j) {
        if (i < 0 || j < 0 || i >= dims || j >= dims) {
            throw std::runtime_error("array_view::swap_dim: Index out of bounds");
        }

        std::swap(data_shape[i], data_shape[j]);
        std::swap(index_offset[i], index_offset[j]);
        std::swap(order[i], order[j]);

        bool tmp_reversed = reversed[i];
        reversed[i] = reversed[j];
        reversed[j] = tmp_reversed;

    }

    void reverse(int i) {
        reversed[i] = !reversed[i];
    }

    size_t size() const {
        return data_size;
    }

    T& operator()(const std::vector<size_t>& indices) {
        assert(indices.size() == static_cast<size_t>(dims));
        size_t linear_index = 0;
        for (int i = 0; i < dims; ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += (reversed[i] ? data_shape[i] - 1 - indices[i] : indices[i]) * index_offset[i];
            }
            else {
                throw std::runtime_error("array_view: Index out of bounds");
            }
        }
        return data[linear_index];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        assert(indices.size() == static_cast<size_t>(dims));
        size_t linear_index = 0;
        for (int i = 0; i < dims; ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += (reversed[i] ? data_shape[i] - 1 - indices[i] : indices[i]) * index_offset[i];
            }
            else {
                throw std::runtime_error("array_view: Index out of bounds");
            }
        }
        return data[linear_index];
    }

    array_view<T> operator[](size_t index) const {
        if (dims == 0) {
            throw std::runtime_error("array_view::operator[]: Array is empty");
        }
        std::vector<size_t> new_shape(data_shape.begin() + 1, data_shape.end());
        std::vector<size_t> new_offset(index_offset.begin() + 1, index_offset.end());
        std::vector<bool> new_reversed(reversed.begin() + 1, reversed.end());
        std::vector<int> new_order(order.begin() + 1, order.end());
        for (int& ord : new_order) {
            if (ord > order[0]) {
                --ord;
            }
        }
        if (index >= data_shape[0]) {
            throw std::runtime_error("array_view: Index out of bounds");
        }
        if (reversed[0]) {
            index = data_shape[0] - 1 - index;
        }
        return array_view<T>(data + index * index_offset[0], new_shape, new_offset, new_reversed, new_order);
    }

    template<typename... Args>
    T& operator()(Args... args) {
        assert(sizeof...(args) == dims);
        std::vector<size_t> indices = { static_cast<size_t>(args)... };
        return (*this)(indices);
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        assert(sizeof...(args) == dims);
        std::vector<size_t> indices = { static_cast<size_t>(args)... };
        return (*this)(indices);
    }

    size_t shape(int i) const {
        return data_shape[i];
    }

    int get_order(int i) const {
        return order[i];
    }

    bool is_reversed(int i) const {
        return reversed[i];
    }

    std::vector<size_t> get_shape() const {
        return data_shape;
    }

    size_t offset(int i) const {
        return index_offset[i];
    }

    int get_dims() const {
        return dims;
    }

    std::vector<int>& get_order() {
        return order;
    }

    const std::vector<int>& get_order() const {
        return order;
    }

    std::vector<bool>& get_reversed() {
        return reversed;
    }

    const std::vector<bool>& get_reversed() const {
        return reversed;
    }

    const std::vector<size_t>& get_offset() const {
        return index_offset;
    }

    bool operator==(const array_view<T>& other) const {
        if (data_size != other.data_size) {
            throw std::runtime_error("array_view: Array sizes do not match");
        }
        for (size_t i = 0; i < data_size; ++i) {
            if (data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }

    T* get_data() {
        return data;
    }
    const T* get_data() const {
        return data;
    }

    bool empty() const {
        return data_size == 0;
    }

    void mimic(const array_view<T>& other, const std::vector<size_t>& new_shape = {}) {
        std::vector<size_t> ns(other.dims); // Original shape
        if (new_shape.empty()) {
            for (int i = 0; i < other.dims; ++i) {
                ns[other.order[i]] = other.data_shape[i];
            }
        } else {
            if (new_shape.size() != other.dims) {
                throw std::runtime_error("array_view::copy_to: New shape size does not match");
            }
            for (int i = 0; i < other.dims; ++i) {
                ns[other.order[i]] = new_shape[i];
            }
        }

        *this = array_view<T>(data, ns);
        reorder(other.order);
        reversed = other.reversed;
    }

    void copy_to(array_view<T>& new_arr, T* output, const std::vector<size_t>& new_shape = {}) const {
        new_arr.mimic(*this, new_shape);
        new_arr.data = output;

        std::vector<size_t> ind(dims);
        for (size_t i = 0; i < data_size; ++i) {
            new_arr(ind) = (*this)(ind);
            ++ind.back();
            int high = dims - 1;
            while (high > 0 && ind[high] == data_shape[high]) {
                ind[high] = 0;
                --high;
                ++ind[high];
            }
        }
    }

    void copy_to(T* output, const std::vector<size_t>& new_shape = {}) const {
        array_view<T> new_arr;
        new_arr.mimic(*this, new_shape);
        new_arr.data = output;

        std::vector<size_t> ind(dims);
        for (size_t i = 0; i < data_size; ++i) {
            new_arr(ind) = (*this)(ind);
            ++ind.back();
            int high = dims - 1;
            while (high > 0 && ind[high] == data_shape[high]) {
                ind[high] = 0;
                --high;
                ++ind[high];
            }
        }
    }

protected:
    void init_offset() {
        if (dims == 0) {
            return;
        }
        index_offset.resize(dims);
        index_offset[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i) {
            index_offset[i] = index_offset[i + 1] * data_shape[i + 1];
        }
    }
    int dims;
    T* data;
    size_t data_size;
    std::vector<int> order;
    std::vector<size_t> data_shape;
    std::vector<size_t> index_offset;
    std::vector<bool> reversed; // whether the index is reversed
    // std::vector<int> ind; // order of indices
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const array_view<T>& arr) {
    os << "[";
    if (arr.dims == 1) {
        for (size_t i = 0; i < arr.data_size; ++i) {
            os << arr.data[i] << " ";
        }
    }
    else {
        for (size_t i = 0; i < arr.data_shape[0]; ++i) {
            os << arr[i];
            if (i < arr.data_shape[0] - 1) {
                os << ", ";
            }
            os << std::endl;
        }
    }
    os << "]";
    return os;
}

template <typename T>
class array {
public:
    array() {}

    array(const array_view<T>& view) {
        size_t sz = view.size();
        data.resize(sz);
        view.copy_to(this->view, data.data());
    }

    array(const std::vector<size_t>& shape) {
        init(shape);
    }

    void init(const std::vector<size_t>& shape) {
        size_t data_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            data_size *= shape[i];
        }
        data.resize(data_size);
        view = array_view<T>(data.data(), shape);
    }

    void rearrange() {
        std::vector<T> new_data(data.size());
        std::vector<size_t> ind(view.dims);
        array_view<T> new_view(new_data.data(), view.data_shape);
        std::vector<array_view<T>> new_views(view.dims), views(view.dims);

        new_views[0] = new_view;
        views[0] = view;

        for (int i = 1; i < view.dims; ++i) {
            new_views[i] = new_views[i - 1][0];
            views[i] = views[i - 1][0];
        }

        while (true) {
            new_views.back()(ind.back()) = views.back()(ind.back());
            ++ind.back();
            int high = view.dims - 1;
            while (high > 0 && ind[high] == view.data_shape[high]) {
                ind[high] = 0;
                --high;
                ++ind[high];
            }

            if (ind[high] == view.data_shape[high]) break;
            

            for (int i = high + 1; i < view.dims; ++i) {
                new_views[i] = new_views[i - 1][ind[i - 1]];
                views[i] = views[i - 1][ind[i - 1]];
            }
        }
        data = new_data;
        view = new_view;
        view.data = data.data();
    }

    operator array_view<T>() {
        return view;
    }

    T& operator()(const std::vector<size_t>& indices) {
        return view(indices);
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return view(indices);
    }

    template<typename... Args>
    T& operator()(Args... args) {
        assert(sizeof...(args) == view.dims);
        std::vector<size_t> indices = { static_cast<size_t>(args)... };
        return (*this)(indices);
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        assert(sizeof...(args) == view.dims);
        std::vector<size_t> indices = { static_cast<size_t>(args)... };
        return (*this)(indices);
    }

    array_view<T> view;
    std::vector<T> data;
};

