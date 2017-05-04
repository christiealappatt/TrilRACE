
namespace stk {
namespace math {

namespace hidden {
static STK_MATH_FORCE_INLINE float CBRT(const float x) { return cbrtf(x); }
static const simd::Float SIGN_MASKf(-0.0);
}

STK_MATH_FORCE_INLINE simd::Float fmadd(const simd::Float& a, const simd::Float& b, const simd::Float& c) {
  return simd::Float(_mm256_add_ps(_mm256_mul_ps(a._data,b._data), c._data));
}

STK_MATH_FORCE_INLINE simd::Float sqrt(const simd::Float& x) {
  return simd::Float(_mm256_sqrt_ps(x._data));
}
  
STK_MATH_FORCE_INLINE simd::Float cbrt(const simd::Float& x) {
#if defined(__INTEL_COMPILER)
  return simd::Float(_mm256_cbrt_ps(x._data));
#else
  simd::Float out;
  for (int i=0; i < simd::nfloats; ++i) {
    out[i] = hidden::CBRT(x[i]);
  }
  return out;
#endif
}

STK_MATH_FORCE_INLINE simd::Float log(const simd::Float& x) {
#if defined(__INTEL_COMPILER) // this log is slightly inaccurate
  return simd::Float(_mm256_log_ps(x._data));
#else
  simd::Float tmp;
  for (int n=0; n < simd::nfloats; ++n) {
    tmp[n] = std::log(x[n]);
  }
  return tmp;
#endif
}

STK_MATH_FORCE_INLINE simd::Float log10(const simd::Float& x) {
  simd::Float tmp;
  for (int n=0; n < simd::nfloats; ++n) {
    tmp[n] = std::log10(x[n]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float exp(const simd::Float& x) {
#if defined(__INTEL_COMPILER)
  return simd::Float(_mm256_exp_ps(x._data));
#else
  simd::Float tmp;
  for (int n=0; n < simd::nfloats; ++n) {
    tmp[n] = std::exp(x[n]);
  }
  return tmp;
#endif
}

STK_MATH_FORCE_INLINE simd::Float pow(const simd::Float& x, const simd::Float& y) {
  simd::Float tmp;
  for (int n=0; n < simd::nfloats; ++n) {
    tmp[n] = std::pow(x[n],y[n]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float pow(const simd::Float& x, float y) {
  simd::Float tmp;
  for (int n=0; n < simd::nfloats; ++n) {
    tmp[n] = std::pow(x[n],y);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float sin(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::sin(a[i]);
  }
  return tmp;
}
 
STK_MATH_FORCE_INLINE simd::Float cos(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::cos(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float tan(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::tan(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float sinh(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::sinh(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float cosh(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::cosh(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float tanh(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::tanh(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float asin(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::asin(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float acos(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::acos(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float atan(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::atan(a[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float atan2(const simd::Float& a, const simd::Float& b) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    tmp[i] = std::atan2(a[i],b[i]);
  }
  return tmp;
}

STK_MATH_FORCE_INLINE simd::Float multiplysign(const simd::Float& x, const simd::Float& y) { // return x times sign of y
  return simd::Float(_mm256_xor_ps(x._data, _mm256_and_ps(hidden::SIGN_MASKf._data, y._data)));
}
  
STK_MATH_FORCE_INLINE simd::Float copysign(const simd::Float& x, const simd::Float& y) { // return abs(x) times sign of y
  return simd::Float(_mm256_xor_ps(_mm256_andnot_ps(hidden::SIGN_MASKf._data, x._data), _mm256_and_ps(hidden::SIGN_MASKf._data, y._data)));
}

STK_MATH_FORCE_INLINE simd::Float abs(const simd::Float& x) {
  return simd::Float(_mm256_andnot_ps(hidden::SIGN_MASKf._data, x._data)); // !sign_mask & x
}

STK_MATH_FORCE_INLINE simd::Float min(const simd::Float& x, const simd::Float& y) {
  return simd::Float(_mm256_min_ps(x._data, y._data));
}

STK_MATH_FORCE_INLINE simd::Float max(const simd::Float& x, const simd::Float& y) {
  return simd::Float(_mm256_max_ps(x._data, y._data));
}
 
STK_MATH_FORCE_INLINE simd::Boolf isnan(const simd::Float& a) {
  simd::Float tmp;
  for (int i=0; i < simd::nfloats; ++i) {
    if ( a[i] != a[i] ) {
      tmp[i] = 1.0;
    } else {
      tmp[i] = 0.0;
    }
  }
  return tmp > simd::Float(0.5f);
}

STK_MATH_FORCE_INLINE simd::Float if_then_else(const simd::Boolf& b, const simd::Float& v1, const simd::Float& v2) {
  return simd::Float( _mm256_add_ps(_mm256_and_ps(b._data, v1._data),_mm256_andnot_ps(b._data, v2._data)) );
}

STK_MATH_FORCE_INLINE simd::Float if_then_else_zero(const simd::Boolf& b, const simd::Float& v) {
  return simd::Float( _mm256_and_ps(b._data, v._data) );
}

STK_MATH_FORCE_INLINE simd::Float if_not_then_else(const simd::Boolf& b, const simd::Float& v1, const simd::Float& v2) {
  return simd::Float( _mm256_add_ps(_mm256_and_ps(b._data, v2._data),_mm256_andnot_ps(b._data, v1._data)) );
}

STK_MATH_FORCE_INLINE simd::Float if_not_then_else_zero(const simd::Boolf& b, const simd::Float& v) {
  return simd::Float( _mm256_andnot_ps(b._data, v._data) );
}


} // namespace math
} // namespace stk
