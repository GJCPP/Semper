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

    array_view() : data(nullptr), data_shape({}), index_offset({}), dims(0), default_value(T()), data_size(0) {}

    array_view(T* _data, const std::vector<size_t>& _shape, T _default_value = T()) 
        : data(_data), data_shape(_shape), index_offset(_shape.size()), dims(_shape.size()), default_value(_default_value), data_size(1) {
        for (size_t i = 0; i < _shape.size(); ++i) {
            data_size *= _shape[i];
        }
        init_offset();
    }

    T& get(size_t index) {
        return data[index];
    }

    size_t size() const {
        return data_size;
    }

    T& operator()(const std::vector<size_t>& indices) {
        assert(indices.size() == dims);
        size_t linear_index = 0;
        for (int i = 0; i < dims; ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += indices[i] * index_offset[i];
            } else {
                return default_value;
            }
        }
        return data[linear_index];
    }

    const T& operator()(std::vector<size_t> indices) const {
        assert(indices.size() == dims);
        size_t linear_index = 0;
        for (int i = 0; i < data_shape.size(); ++i) {
            if (indices[i] < data_shape[i]) {
                linear_index += indices[i] * index_offset[i];
            } else {
                return default_value;
            }
        }
        return data[linear_index];
    }

    array_view<T> operator[](size_t index) const {
        std::vector<size_t> new_shape(data_shape.begin() + 1, data_shape.end());
        return array_view<T>(data + index * index_offset[0], new_shape);
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
    T default_value;
    T* data;
    size_t data_size;
    std::vector<size_t> data_shape;
    std::vector<size_t> index_offset;
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