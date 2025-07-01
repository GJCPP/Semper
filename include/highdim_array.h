#pragma once
#include <cassert>
#include <array>
#include <vector>

template<typename T>
class array_view;

template<typename T>
std::ostream& operator<<(std::ostream& os, const array_view<T>& arr);

template<typename T>
class array_view {
public:
    friend std::ostream& operator<< <T>(std::ostream& os, const array_view<T>& arr);

    array_view() : data(nullptr), data_shape({}), index_offset({}), dims(0), data_size(0), reversed({}) {}

    array_view(T* _data, const std::vector<size_t>& _shape, const std::vector<size_t>& _offset = {}, const std::vector<bool>& _reversed = {}) 
        : data(_data), data_shape(_shape), index_offset(_shape.size()), dims(_shape.size()), data_size(1) {
        for (size_t i = 0; i < _shape.size(); ++i) {
            data_size *= _shape[i];
        }
        if (_offset.size() == 0) {
            init_offset();
        } else {
            index_offset = _offset;
        }
        if (_reversed.size() == 0) {
            reversed = std::vector<bool>(_shape.size(), false);
        } else {
            reversed = _reversed;
        }
    }

    T& get(size_t index) {
        return data[index];
    }

    const T& get(size_t index) const {
        return data[index];
    }

    void reorder(const std::vector<int>& ind) {
        if (ind.size() != dims) {
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
    }

    void swap_dim(int i, int j) {
        std::swap(data_shape[i], data_shape[j]);
        std::swap(index_offset[i], index_offset[j]);
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
        assert(indices.size() == dims);
        size_t linear_index = 0;
        for (int i = 0; i < dims; ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += (reversed[i] ? data_shape[i] - 1 - indices[i] : indices[i]) * index_offset[i];
            } else {
                throw std::runtime_error("array_view: Index out of bounds");
            }
        }
        return data[linear_index];
    }

    const T& operator()(std::vector<size_t> indices) const {
        assert(indices.size() == dims);
        size_t linear_index = 0;
        for (int i = 0; i < dims; ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += (reversed[i] ? data_shape[i] - 1 - indices[i] : indices[i]) * index_offset[i];
            } else {
                throw std::runtime_error("array_view: Index out of bounds");
            }
        }
        return data[linear_index];
    }

    array_view<T> operator[](size_t index) const {
        std::vector<size_t> new_shape(data_shape.begin() + 1, data_shape.end());
        std::vector<size_t> new_offset(index_offset.begin() + 1, index_offset.end());
        std::vector<bool> new_reversed(reversed.begin() + 1, reversed.end());
        if (index >= data_shape[0]) {
            throw std::runtime_error("array_view: Index out of bounds");
        }
        if (reversed[0]) {
            index = data_shape[0] - 1 - index;
        }
        return array_view<T>(data + index * index_offset[0], new_shape, new_offset, new_reversed);
    }

    template<typename... Args>
    T& operator()(Args... args) {
        assert(sizeof...(args) == dims);
        std::vector<size_t> indices = {static_cast<size_t>(args)...};
        return (*this)(indices);
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        assert(sizeof...(args) == dims);
        std::vector<size_t> indices = {static_cast<size_t>(args)...};
        return (*this)(indices);
    }

    size_t shape(int i) const {
        return data_shape[i];
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

protected:
    void init_offset() {
        if (dims == 0) {
            return;
        }
        index_offset[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i) {
            index_offset[i] = index_offset[i + 1] * data_shape[i + 1];
        }
    }
    int dims;
    T* data;
    size_t data_size;
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
    } else {
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