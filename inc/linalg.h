#ifndef LINALG_H_
#define LINALG_H_

#include <Eigen/Dense>

#include <svd3/svd3_cuda/svd3_cuda.h>

namespace linalg {
    __host__ __device__ void svd3(const Eigen::Matrix3f& a, Eigen::Matrix3f& u, Eigen::Matrix3f& s, Eigen::Matrix3f& v) {
        svd(a(0), a(3), a(6), a(1), a(4), a(7), a(2), a(5), a(8),
            u(0), u(3), u(6), u(1), u(4), u(7), u(2), u(5), u(8),
            s(0), s(3), s(6), s(1), s(4), s(7), s(2), s(5), s(8),
            v(0), v(3), v(6), v(1), v(4), v(7), v(2), v(5), v(8));
    }

    __host__ __device__ float determinant(const Eigen::Matrix3f& a) {
        return a(0) * (a(4) * a(8) - a(7) * a(5)) -
               a(3) * (a(1) * a(8) - a(7) * a(2)) +
               a(6) * (a(1) * a(5) - a(4) * a(2));
    }

    __host__ __device__ Eigen::Matrix3f inverse(const Eigen::Matrix3f& a) {
        float inv_det = 1.0 / determinant(a);

        Eigen::Matrix3f b;

        b(0) = inv_det * (a(4) * a(8) - a(5) * a(7));
        b(1) = inv_det * (a(2) * a(7) - a(1) * a(8));
        b(2) = inv_det * (a(1) * a(5) - a(2) * a(4));
        b(3) = inv_det * (a(5) * a(6) - a(3) * a(8));
        b(4) = inv_det * (a(0) * a(8) - a(2) * a(6));
        b(5) = inv_det * (a(2) * a(3) - a(0) * a(5));
        b(6) = inv_det * (a(3) * a(7) - a(4) * a(6));
        b(7) = inv_det * (a(1) * a(6) - a(0) * a(7));
        b(8) = inv_det * (a(0) * a(4) - a(1) * a(3));

        return b;
    }

    __host__ __device__ Eigen::Matrix3f cofactor(const Eigen::Matrix3f& a) {
        // http://www.mathwords.com/c/cofactor_matrix.htm
        Eigen::Matrix3f b;

        b(0) = a(4) * a(8) - a(5) * a(7);
        b(1) = a(5) * a(6) - a(3) * a(8);
        b(2) = a(3) * a(7) - a(4) * a(6);
        b(3) = a(2) * a(7) - a(1) * a(8);
        b(4) = a(0) * a(8) - a(2) * a(6);
        b(5) = a(1) * a(6) - a(0) * a(7);
        b(6) = a(1) * a(5) - a(2) * a(4);
        b(7) = a(2) * a(3) - a(0) * a(5);
        b(8) = a(0) * a(4) - a(1) * a(3);

        return b;
    }

    __host__ __device__ Eigen::Vector3f solve(const Eigen::Matrix3f& a, const Eigen::Vector3f& b) {
        return inverse(a) * b;
    }
}

#endif  // LINALG_H_
