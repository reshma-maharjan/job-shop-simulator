#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <multidimensional_array.hpp>

namespace nb = nanobind;

template<typename T, std::size_t NDim>
nb::ndarray<T, nb::shape<nb::any>, nb::device::cpu> wrap_multi_dim_array(MultiDimensionalArray<T, NDim>& arr) {
    return nb::ndarray<T, nb::shape<nb::any>, nb::device::cpu>(
            arr.data_ptr(),
                    NDim,
                    arr.shape().data(),
                    arr.strides().data()
    );
}

template<typename T, std::size_t NDim>
void bind_multi_dim_array(nb::module_ &m, const char* name) {
    nb::class_<MultiDimensionalArray<T, NDim>>(m, name)
            .def(nb::init<const std::array<std::size_t, NDim>&>())
            .def("__array__", [](MultiDimensionalArray<T, NDim>& arr) {
                return wrap_multi_dim_array(arr);
            })
            .def("shape", &MultiDimensionalArray<T, NDim>::shape)
            .def("size", &MultiDimensionalArray<T, NDim>::size)
            .def("fill", &MultiDimensionalArray<T, NDim>::fill);
}

NB_MODULE(multi_dim_array_ext, m) {
bind_multi_dim_array<float, 1>(m, "MultiDimArray1f");
bind_multi_dim_array<float, 2>(m, "MultiDimArray2f");
bind_multi_dim_array<float, 3>(m, "MultiDimArray3f");
bind_multi_dim_array<double, 1>(m, "MultiDimArray1d");
bind_multi_dim_array<double, 2>(m, "MultiDimArray2d");
bind_multi_dim_array<double, 3>(m, "MultiDimArray3d");
bind_multi_dim_array<int, 1>(m, "MultiDimArray1i");
bind_multi_dim_array<int, 2>(m, "MultiDimArray2i");
bind_multi_dim_array<int, 3>(m, "MultiDimArray3i");
}