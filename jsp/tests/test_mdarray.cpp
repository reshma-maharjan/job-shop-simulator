#include <gtest/gtest.h>
#include "multidimensional_array.hpp"
#include <functional>

class MultiDimensionalArrayTest : public ::testing::Test {
protected:
    MultiDimensionalArray<int, 3> arr3d{std::array<std::size_t, 3>{2, 3, 4}};
    MultiDimensionalArray<double, 2> arr2d{std::array<std::size_t, 2>{3, 3}};
};

TEST_F(MultiDimensionalArrayTest, ConstructorAndDimensions) {
EXPECT_EQ(arr3d.getDimensions(), (std::array<std::size_t, 3>{2, 3, 4}));
EXPECT_EQ(arr2d.getDimensions(), (std::array<std::size_t, 2>{3, 3}));
}

TEST_F(MultiDimensionalArrayTest, ElementAccessAndModification) {
arr3d(0, 0, 0) = 1;
arr3d(1, 2, 3) = 42;

EXPECT_EQ(arr3d(0, 0, 0), 1);
EXPECT_EQ(arr3d(1, 2, 3), 42);

arr2d(0, 0) = 3.14;
arr2d(2, 2) = 2.718;

EXPECT_DOUBLE_EQ(arr2d(0, 0), 3.14);
EXPECT_DOUBLE_EQ(arr2d(2, 2), 2.718);
}

TEST_F(MultiDimensionalArrayTest, IteratorSupport) {
int value = 0;
for (auto& elem : arr3d) {
elem = value++;
}

EXPECT_EQ(arr3d(0, 0, 0), 0);
EXPECT_EQ(arr3d(0, 0, 1), 1);
EXPECT_EQ(arr3d(1, 2, 3), 23);

double sum = 0.0;
for (const auto& elem : arr2d) {
sum += elem;
}
EXPECT_DOUBLE_EQ(sum, 0.0);
}

TEST_F(MultiDimensionalArrayTest, Fill) {
arr3d.fill(42);
for (const auto& elem : arr3d) {
EXPECT_EQ(elem, 42);
}

arr2d.fill(3.14);
for (const auto& elem : arr2d) {
EXPECT_DOUBLE_EQ(elem, 3.14);
}
}

TEST_F(MultiDimensionalArrayTest, Transform) {
    arr3d.fill(2);
    arr3d.transform(std::function<int(const int&)>([](const int& x) { return x * x; }));
    for (const auto& elem : arr3d) {
        EXPECT_EQ(elem, 4);
    }

    arr2d.fill(2.0);
    arr2d.transform(std::function<double(const double&)>([](const double& x) { return x + 1.0; }));
    for (const auto& elem : arr2d) {
        EXPECT_DOUBLE_EQ(elem, 3.0);
    }
}

TEST_F(MultiDimensionalArrayTest, SizeAndEmpty) {
EXPECT_EQ(arr3d.size(), 24);
EXPECT_EQ(arr2d.size(), 9);
EXPECT_FALSE(arr3d.empty());
EXPECT_FALSE(arr2d.empty());

MultiDimensionalArray<int, 1> empty_arr{std::array<std::size_t, 1>{0}};
EXPECT_TRUE(empty_arr.empty());
EXPECT_EQ(empty_arr.size(), 0);
}

TEST_F(MultiDimensionalArrayTest, MoveSemantics) {
MultiDimensionalArray<std::string, 2> arr1{std::array<std::size_t, 2>{2, 2}};
arr1(0, 0) = "Hello";
arr1(1, 1) = "World";

MultiDimensionalArray<std::string, 2> arr2 = std::move(arr1);

EXPECT_EQ(arr2(0, 0), "Hello");
EXPECT_EQ(arr2(1, 1), "World");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}