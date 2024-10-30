#pragma once
#include <array>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>
#include <functional>

template<typename T, std::size_t NDim>
class MultiDimensionalArray {
private:
    alignas(64) std::vector<T> data;  // Aligned for better cache performance
    alignas(64) std::array<std::size_t, NDim> dimensions;
    alignas(64) std::array<std::size_t, NDim> stride_array;

    void calculateStrides() noexcept {
        stride_array[NDim - 1] = 1;
        for (int i = NDim - 2; i >= 0; --i) {
            stride_array[i] = stride_array[i + 1] * dimensions[i + 1];
        }
    }

    template<typename... Indices>
    [[nodiscard]] inline std::size_t calculateIndex(Indices... indices) const noexcept {
        std::array<std::size_t, NDim> idx{static_cast<std::size_t>(indices)...};
        std::size_t index = 0;
        for (std::size_t i = 0; i < NDim; ++i) {
            index += idx[i] * stride_array[i];
        }
        return index;
    }

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    explicit MultiDimensionalArray(const std::array<std::size_t, NDim>& dims) noexcept
            : dimensions(dims) {
        calculateStrides();
        data.resize(std::accumulate(dims.begin(), dims.end(), std::size_t(1), std::multiplies<>()));
    }

    MultiDimensionalArray(const MultiDimensionalArray& other) = default;
    MultiDimensionalArray(MultiDimensionalArray&& other) noexcept = default;
    MultiDimensionalArray& operator=(const MultiDimensionalArray& other) = default;
    MultiDimensionalArray& operator=(MultiDimensionalArray&& other) noexcept = default;

    template<typename... Indices>
    [[nodiscard]] inline const T& operator()(Indices... indices) const noexcept {
        return data[calculateIndex(indices...)];
    }

    template<typename... Indices>
    [[nodiscard]] inline T& operator()(Indices... indices) noexcept {
        return data[calculateIndex(indices...)];
    }

    [[nodiscard]] inline const std::array<std::size_t, NDim>& getDimensions() const noexcept {
        return dimensions;
    }

    [[nodiscard]] inline iterator begin() noexcept { return data.begin(); }
    [[nodiscard]] inline iterator end() noexcept { return data.end(); }
    [[nodiscard]] inline const_iterator begin() const noexcept { return data.begin(); }
    [[nodiscard]] inline const_iterator end() const noexcept { return data.end(); }
    [[nodiscard]] inline const_iterator cbegin() const noexcept { return data.cbegin(); }
    [[nodiscard]] inline const_iterator cend() const noexcept { return data.cend(); }
    [[nodiscard]] inline size_type size() const noexcept { return data.size(); }
    [[nodiscard]] inline bool empty() const noexcept { return data.empty(); }

    inline void fill(const T& value) noexcept {
        std::fill(data.begin(), data.end(), value);
    }

    inline void transform(const std::function<T(const T&)>& func) {
        std::transform(data.begin(), data.end(), data.begin(), func);
    }

    [[nodiscard]] inline pointer data_ptr() noexcept { return data.data(); }
    [[nodiscard]] inline const_pointer data_ptr() const noexcept { return data.data(); }
    [[nodiscard]] inline const std::array<std::size_t, NDim>& shape() const noexcept { return dimensions; }
    [[nodiscard]] inline const std::array<std::size_t, NDim>& strides() const noexcept { return stride_array; }

    // Print function (simplified for brevity)
    void print() const {
        // Implementation left as an exercise
    }
};