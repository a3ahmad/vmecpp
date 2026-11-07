// SPDX-License-Identifier: MIT
#ifndef VMECPP_PRIVATE_RAGGED_SHAPE_H_
#define VMECPP_PRIVATE_RAGGED_SHAPE_H_
#if defined(__CUDACC__)
#define VMECPP_RAGGED_HD __host__ __device__
#else
#define VMECPP_RAGGED_HD
#endif
struct RaggedShape {
  int first_count_;
  int second_count_;
  VMECPP_RAGGED_HD int Count() const { return first_count_ + second_count_; }
  VMECPP_RAGGED_HD int Component(int index) const {
    return index < first_count_ ? 0 : 1;
  }
  VMECPP_RAGGED_HD int Offset(int component) const {
    return component == 0 ? 0 : first_count_;
  }
  VMECPP_RAGGED_HD int Length(int component) const {
    return component == 0 ? first_count_ : second_count_;
  }
  VMECPP_RAGGED_HD int Local(int index) const {
    return index - Offset(Component(index));
  }
};
#undef VMECPP_RAGGED_HD
#endif
