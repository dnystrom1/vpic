#ifndef _v4_portable_h_
#define _v4_portable_h_

#ifndef IN_v4_h
#error "Do not include v4_portable.h directly; use v4.h"
#endif

#define V4_ACCELERATION
#define V4_PORTABLE_ACCELERATION

#include <math.h>

#ifndef ALIGNED
#define ALIGNED(n)
#endif

namespace v4
{
  class v4;
  class v4int;
  class v4float;

  ////////////////
  // v4 base class

  class v4
  {
    friend class v4int;
    friend class v4float;

    // v4 miscellenous friends

    friend inline int any( const v4 &a );
    friend inline int all( const v4 &a );

    template<int n>
    friend inline v4 splat( const v4 &a );

    template<int i0, int i1, int i2, int i3>
    friend inline v4 shuffle( const v4 &a );

    friend inline void swap( v4 &a, v4 &b );
    friend inline void transpose( v4 &a0, v4 &a1, v4 &a2, v4 &a3 );

    // v4int miscellaneous friends

    friend inline v4    czero( const v4int &c, const v4 &a );
    friend inline v4 notczero( const v4int &c, const v4 &a );
    friend inline v4 merge( const v4int &c, const v4 &a, const v4 &b );

    // v4 memory manipulation friends

    friend inline void   load_4x1( const void * ALIGNED(16) p, v4 &a );
    friend inline void  store_4x1( const v4 &a, void * ALIGNED(16) p );
    friend inline void stream_4x1( const v4 &a, void * ALIGNED(16) p );
    friend inline void   copy_4x1( void * ALIGNED(16) dst,
                                   const void * ALIGNED(16) src );
    friend inline void   swap_4x1( void * ALIGNED(16) a, void * ALIGNED(16) b );

    // v4 transposed memory manipulation friends
    // Note: Half aligned values are permissible in the 4x2_tr variants!

    friend inline void load_4x1_tr( const void *a0, const void *a1,
                                    const void *a2, const void *a3,
                                    v4 &a );
    friend inline void load_4x2_tr( const void * ALIGNED(8) a0,
                                    const void * ALIGNED(8) a1,
                                    const void * ALIGNED(8) a2,
                                    const void * ALIGNED(8) a3,
                                    v4 &a, v4 &b );
    friend inline void load_4x3_tr( const void * ALIGNED(16) a0,
                                    const void * ALIGNED(16) a1,
                                    const void * ALIGNED(16) a2,
                                    const void * ALIGNED(16) a3,
                                    v4 &a, v4 &b, v4 &c );
    friend inline void load_4x4_tr( const void * ALIGNED(16) a0,
                                    const void * ALIGNED(16) a1,
                                    const void * ALIGNED(16) a2,
                                    const void * ALIGNED(16) a3,
                                    v4 &a, v4 &b, v4 &c, v4 &d );

    friend inline void store_4x1_tr( const v4 &a,
                                     void *a0, void *a1, void *a2, void *a3 );
    friend inline void store_4x2_tr( const v4 &a, const v4 &b,
                                     void * ALIGNED(8) a0,
                                     void * ALIGNED(8) a1,
                                     void * ALIGNED(8) a2,
                                     void * ALIGNED(8) a3 );
    friend inline void store_4x3_tr( const v4 &a, const v4 &b, const v4 &c,
                                     void * ALIGNED(16) a0,
                                     void * ALIGNED(16) a1,
                                     void * ALIGNED(16) a2,
                                     void * ALIGNED(16) a3 );
    friend inline void store_4x4_tr( const v4 &a, const v4 &b,
                                     const v4 &c, const v4 &d,
                                     void * ALIGNED(16) a0,
                                     void * ALIGNED(16) a1,
                                     void * ALIGNED(16) a2,
                                     void * ALIGNED(16) a3 );

  protected:

    union
    {
      int i[4];
      float f[4];
    };

  public:

    #pragma forceinline recursive
    v4() {}                    // Default constructor

    #pragma forceinline recursive
    v4( const v4 &a )          // Copy constructor
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	i[j] = a.i[j];
    }

    #pragma forceinline recursive
    ~v4() {}                   // Default destructor
  };

  // v4 miscellaneous functions

  #pragma forceinline recursive
  inline int any( const v4 &a )
  {
    return a.i[0] || a.i[1] || a.i[2] || a.i[3];
  }

  #pragma forceinline recursive
  inline int all( const v4 &a )
  {
    return a.i[0] && a.i[1] && a.i[2] && a.i[3];
  }

  #pragma forceinline recursive
  template<int n>
  inline v4 splat( const v4 & a )
  {
    v4 b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = a.i[n];

    return b;
  }

  #pragma forceinline recursive
  template<int i0, int i1, int i2, int i3>
  inline v4 shuffle( const v4 & a )
  {
    v4 b;

    b.i[0] = a.i[i0];
    b.i[1] = a.i[i1];
    b.i[2] = a.i[i2];
    b.i[3] = a.i[i3];

    return b;
  }

# define sw(x,y) x^=y, y^=x, x^=y

  #pragma forceinline recursive
  inline void swap( v4 &a, v4 &b )
  { 
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      sw( a.i[j], b.i[j] );
  }

  #pragma forceinline recursive
  inline void transpose( v4 &a0, v4 &a1, v4 &a2, v4 &a3 )
  {
    sw( a0.i[1],a1.i[0] ); sw( a0.i[2],a2.i[0] ); sw( a0.i[3],a3.i[0] );
                           sw( a1.i[2],a2.i[1] ); sw( a1.i[3],a3.i[1] );
                                                  sw( a2.i[3],a3.i[2] );
  }

# undef sw

  // v4 memory manipulation functions
  
  #pragma forceinline recursive
  inline void load_4x1( const void * ALIGNED(16) p, v4 &a )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      a.i[j] = ((const int * ALIGNED(16))p)[j];
  }

  #pragma forceinline recursive
  inline void store_4x1( const v4 &a, void * ALIGNED(16) p )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      ((int * ALIGNED(16))p)[j] = a.i[j];
  }

  #pragma forceinline recursive
  inline void stream_4x1( const v4 &a, void * ALIGNED(16) p )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      ((int * ALIGNED(16))p)[j] = a.i[j];
  }

  #pragma forceinline recursive
  inline void clear_4x1( void * ALIGNED(16) p )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      ((int * ALIGNED(16))p)[j] = 0;
  }

  // FIXME: Ordering semantics
  #pragma forceinline recursive
  inline void copy_4x1( void * ALIGNED(16) dst,
                        const void * ALIGNED(16) src )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      ((int * ALIGNED(16))dst)[j] = ((const int * ALIGNED(16))src)[j];
  }

  #pragma forceinline recursive
  inline void swap_4x1( void * ALIGNED(16) a, void * ALIGNED(16) b )
  {
    int t;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
    {
      t = ((int * ALIGNED(16))a)[j];
      ((int * ALIGNED(16))a)[j] = ((int * ALIGNED(16))b)[j];
      ((int * ALIGNED(16))b)[j] = t;
    }
  }

  // v4 transposed memory manipulation functions

  #pragma forceinline recursive
  inline void load_4x1_tr( const void *a0, const void *a1,
                           const void *a2, const void *a3,
			   v4 &a )
  {
    a.i[0] = ((const int *)a0)[0];
    a.i[1] = ((const int *)a1)[0];
    a.i[2] = ((const int *)a2)[0];
    a.i[3] = ((const int *)a3)[0];
  }

  #pragma forceinline recursive
  inline void load_4x2_tr( const void * ALIGNED(8) a0,
                           const void * ALIGNED(8) a1,
                           const void * ALIGNED(8) a2,
                           const void * ALIGNED(8) a3,
                           v4 &a, v4 &b )
  {
    a.i[0] = ((const int * ALIGNED(8))a0)[0];
    b.i[0] = ((const int * ALIGNED(8))a0)[1];

    a.i[1] = ((const int * ALIGNED(8))a1)[0];
    b.i[1] = ((const int * ALIGNED(8))a1)[1];

    a.i[2] = ((const int * ALIGNED(8))a2)[0];
    b.i[2] = ((const int * ALIGNED(8))a2)[1];

    a.i[3] = ((const int * ALIGNED(8))a3)[0];
    b.i[3] = ((const int * ALIGNED(8))a3)[1];
  }

  #pragma forceinline recursive
  inline void load_4x3_tr( const void * ALIGNED(16) a0,
                           const void * ALIGNED(16) a1,
                           const void * ALIGNED(16) a2,
                           const void * ALIGNED(16) a3,
                           v4 &a, v4 &b, v4 &c )
  {
    a.i[0] = ((const int * ALIGNED(16))a0)[0];
    b.i[0] = ((const int * ALIGNED(16))a0)[1];
    c.i[0] = ((const int * ALIGNED(16))a0)[2];

    a.i[1] = ((const int * ALIGNED(16))a1)[0];
    b.i[1] = ((const int * ALIGNED(16))a1)[1];
    c.i[1] = ((const int * ALIGNED(16))a1)[2];

    a.i[2] = ((const int * ALIGNED(16))a2)[0];
    b.i[2] = ((const int * ALIGNED(16))a2)[1];
    c.i[2] = ((const int * ALIGNED(16))a2)[2];

    a.i[3] = ((const int * ALIGNED(16))a3)[0];
    b.i[3] = ((const int * ALIGNED(16))a3)[1];
    c.i[3] = ((const int * ALIGNED(16))a3)[2]; 
  }

  #pragma forceinline recursive
  inline void load_4x4_tr( const void * ALIGNED(16) a0,
                           const void * ALIGNED(16) a1,
                           const void * ALIGNED(16) a2,
                           const void * ALIGNED(16) a3,
                           v4 &a, v4 &b, v4 &c, v4 &d )
  {
    a.i[0] = ((const int * ALIGNED(16))a0)[0];
    b.i[0] = ((const int * ALIGNED(16))a0)[1];
    c.i[0] = ((const int * ALIGNED(16))a0)[2];
    d.i[0] = ((const int * ALIGNED(16))a0)[3];

    a.i[1] = ((const int * ALIGNED(16))a1)[0];
    b.i[1] = ((const int * ALIGNED(16))a1)[1];
    c.i[1] = ((const int * ALIGNED(16))a1)[2];
    d.i[1] = ((const int * ALIGNED(16))a1)[3];

    a.i[2] = ((const int * ALIGNED(16))a2)[0];
    b.i[2] = ((const int * ALIGNED(16))a2)[1];
    c.i[2] = ((const int * ALIGNED(16))a2)[2];
    d.i[2] = ((const int * ALIGNED(16))a2)[3];

    a.i[3] = ((const int * ALIGNED(16))a3)[0];
    b.i[3] = ((const int * ALIGNED(16))a3)[1];
    c.i[3] = ((const int * ALIGNED(16))a3)[2];
    d.i[3] = ((const int * ALIGNED(16))a3)[3];
  }

  #pragma forceinline recursive
  inline void store_4x1_tr( const v4 &a,
                            void *a0, void *a1, void *a2, void *a3 )
  {
    ((int *)a0)[0] = a.i[0];
    ((int *)a1)[0] = a.i[1];
    ((int *)a2)[0] = a.i[2];
    ((int *)a3)[0] = a.i[3];
  }

  #pragma forceinline recursive
  inline void store_4x2_tr( const v4 &a, const v4 &b,
                            void * ALIGNED(8) a0, void * ALIGNED(8) a1,
                            void * ALIGNED(8) a2, void * ALIGNED(8) a3 )
  {
    ((int * ALIGNED(8))a0)[0] = a.i[0];
    ((int * ALIGNED(8))a0)[1] = b.i[0];

    ((int * ALIGNED(8))a1)[0] = a.i[1];
    ((int * ALIGNED(8))a1)[1] = b.i[1];

    ((int * ALIGNED(8))a2)[0] = a.i[2];
    ((int * ALIGNED(8))a2)[1] = b.i[2];

    ((int * ALIGNED(8))a3)[0] = a.i[3];
    ((int * ALIGNED(8))a3)[1] = b.i[3];
  }

  #pragma forceinline recursive
  inline void store_4x3_tr( const v4 &a, const v4 &b, const v4 &c,
                            void * ALIGNED(16) a0, void * ALIGNED(16) a1,
                            void * ALIGNED(16) a2, void * ALIGNED(16) a3 )
  {
    ((int * ALIGNED(16))a0)[0] = a.i[0];
    ((int * ALIGNED(16))a0)[1] = b.i[0];
    ((int * ALIGNED(16))a0)[2] = c.i[0];

    ((int * ALIGNED(16))a1)[0] = a.i[1];
    ((int * ALIGNED(16))a1)[1] = b.i[1];
    ((int * ALIGNED(16))a1)[2] = c.i[1];

    ((int * ALIGNED(16))a2)[0] = a.i[2];
    ((int * ALIGNED(16))a2)[1] = b.i[2];
    ((int * ALIGNED(16))a2)[2] = c.i[2];

    ((int * ALIGNED(16))a3)[0] = a.i[3];
    ((int * ALIGNED(16))a3)[1] = b.i[3];
    ((int * ALIGNED(16))a3)[2] = c.i[3];
  }

  #pragma forceinline recursive
  inline void store_4x4_tr( const v4 &a, const v4 &b, const v4 &c, const v4 &d,
                            void * ALIGNED(16) a0, void * ALIGNED(16) a1,
                            void * ALIGNED(16) a2, void * ALIGNED(16) a3 )
  {
    ((int * ALIGNED(16))a0)[0] = a.i[0];
    ((int * ALIGNED(16))a0)[1] = b.i[0];
    ((int * ALIGNED(16))a0)[2] = c.i[0];
    ((int * ALIGNED(16))a0)[3] = d.i[0];

    ((int * ALIGNED(16))a1)[0] = a.i[1];
    ((int * ALIGNED(16))a1)[1] = b.i[1];
    ((int * ALIGNED(16))a1)[2] = c.i[1];
    ((int * ALIGNED(16))a1)[3] = d.i[1];

    ((int * ALIGNED(16))a2)[0] = a.i[2];
    ((int * ALIGNED(16))a2)[1] = b.i[2];
    ((int * ALIGNED(16))a2)[2] = c.i[2];
    ((int * ALIGNED(16))a2)[3] = d.i[2];

    ((int * ALIGNED(16))a3)[0] = a.i[3];
    ((int * ALIGNED(16))a3)[1] = b.i[3];
    ((int * ALIGNED(16))a3)[2] = c.i[3];
    ((int * ALIGNED(16))a3)[3] = d.i[3];
  }

  //////////////
  // v4int class

  class v4int : public v4
  {
    // v4int prefix unary operator friends

    friend inline v4int operator  +( const v4int & a );
    friend inline v4int operator  -( const v4int & a );
    friend inline v4int operator  ~( const v4int & a );
    friend inline v4int operator  !( const v4int & a );
    // Note: Referencing (*) and dereferencing (&) apply to the whole vector

    // v4int prefix increment / decrement operator friends

    friend inline v4int operator ++( v4int & a );
    friend inline v4int operator --( v4int & a );

    // v4int postfix increment / decrement operator friends

    friend inline v4int operator ++( v4int & a, int );
    friend inline v4int operator --( v4int & a, int );

    // v4int binary operator friends

    friend inline v4int operator  +( const v4int &a, const v4int &b );
    friend inline v4int operator  -( const v4int &a, const v4int &b );
    friend inline v4int operator  *( const v4int &a, const v4int &b );
    friend inline v4int operator  /( const v4int &a, const v4int &b );
    friend inline v4int operator  %( const v4int &a, const v4int &b );
    friend inline v4int operator  ^( const v4int &a, const v4int &b );
    friend inline v4int operator  &( const v4int &a, const v4int &b );
    friend inline v4int operator  |( const v4int &a, const v4int &b );
    friend inline v4int operator <<( const v4int &a, const v4int &b );
    friend inline v4int operator >>( const v4int &a, const v4int &b );

    // v4int logical operator friends

    friend inline v4int operator  <( const v4int &a, const v4int &b );
    friend inline v4int operator  >( const v4int &a, const v4int &b );
    friend inline v4int operator ==( const v4int &a, const v4int &b );
    friend inline v4int operator !=( const v4int &a, const v4int &b );
    friend inline v4int operator <=( const v4int &a, const v4int &b );
    friend inline v4int operator >=( const v4int &a, const v4int &b );
    friend inline v4int operator &&( const v4int &a, const v4int &b );
    friend inline v4int operator ||( const v4int &a, const v4int &b );

    // v4int miscellaneous friends

    friend inline v4int abs( const v4int &a );
    friend inline v4    czero( const v4int &c, const v4 &a );
    friend inline v4 notczero( const v4int &c, const v4 &a );
    // FIXME: cswap, notcswap!
    friend inline v4 merge( const v4int &c, const v4 &t, const v4 &f );

    // v4float unary operator friends

    friend inline v4int operator  !( const v4float & a ); 

    // v4float logical operator friends

    friend inline v4int operator  <( const v4float &a, const v4float &b );
    friend inline v4int operator  >( const v4float &a, const v4float &b );
    friend inline v4int operator ==( const v4float &a, const v4float &b );
    friend inline v4int operator !=( const v4float &a, const v4float &b );
    friend inline v4int operator <=( const v4float &a, const v4float &b );
    friend inline v4int operator >=( const v4float &a, const v4float &b );
    friend inline v4int operator &&( const v4float &a, const v4float &b );
    friend inline v4int operator ||( const v4float &a, const v4float &b );

    // v4float miscellaneous friends

    friend inline v4float clear_bits(  const v4int &m, const v4float &a );
    friend inline v4float set_bits(    const v4int &m, const v4float &a );
    friend inline v4float toggle_bits( const v4int &m, const v4float &a );

  public:

    // v4int constructors / destructors

    #pragma forceinline recursive
    v4int() {}                                // Default constructor

    #pragma forceinline recursive
    v4int( const v4int &a )                   // Copy constructor
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	i[j] = a.i[j];
    }

    #pragma forceinline recursive
    v4int( const v4 &a )                      // Init from mixed
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	i[j] = a.i[j];
    }

    #pragma forceinline recursive
    v4int( int a )                            // Init from scalar
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	i[j] = a;
    }

    #pragma forceinline recursive
    v4int( int i0, int i1, int i2, int i3 )   // Init from scalars
    {
      i[0] = i0; i[1] = i1; i[2] = i2; i[3] = i3;
    }

    #pragma forceinline recursive
    ~v4int() {}                               // Destructor

    // v4int assignment operators
  
#   define ASSIGN(op)			          \
    _Pragma( "forceinline recursive" )		  \
    inline v4int &operator op( const v4int &b )   \
    {						  \
      _Pragma( "omp simd" )                       \
      for( int j = 0; j < 4; j++ )                \
        i[j] op b.i[j];                           \
      return *this;                               \
    }

    ASSIGN( =)
    ASSIGN(+=)
    ASSIGN(-=)
    ASSIGN(*=)
    ASSIGN(/=)
    ASSIGN(%=)
    ASSIGN(^=)
    ASSIGN(&=)
    ASSIGN(|=)
    ASSIGN(<<=)
    ASSIGN(>>=)

#   undef ASSIGN

    // v4int member access operator

    #pragma forceinline recursive
    inline int &operator []( int n )
    {
      return i[n];
    }

    #pragma forceinline recursive
    inline int  operator ()( int n )
    {
      return i[n];
    }
  };

  // v4int prefix unary operators

# define PREFIX_UNARY(op)                       \
  _Pragma( "forceinline recursive" )		\
  inline v4int operator op( const v4int & a )   \
  {						\
    v4int b;                                    \
    _Pragma( "omp simd" )                       \
    for( int j = 0; j < 4; j++ )                \
      b.i[j] = (op a.i[j]);                     \
    return b;                                   \
  }

  PREFIX_UNARY(+)
  PREFIX_UNARY(-)

  #pragma forceinline recursive
  inline v4int operator !( const v4int & a )
  {
    v4int b;
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = -(!a.i[j]);
    return b;
  }

  PREFIX_UNARY(~)

# undef PREFIX_UNARY

  // v4int prefix increment / decrement

# define PREFIX_INCDEC(op)                      \
  _Pragma( "forceinline recursive" )		\
  inline v4int operator op( v4int & a )         \
  {						\
    v4int b;                                    \
    _Pragma( "omp simd" )                       \
    for( int j = 0; j < 4; j++ )                \
      b.i[j] = (op a.i[j]);                     \
    return b;                                   \
  }

  PREFIX_INCDEC(++)
  PREFIX_INCDEC(--)

# undef PREFIX_INCDEC

  // v4int postfix increment / decrement

# define POSTFIX_INCDEC(op)                    \
  _Pragma( "forceinline recursive" )	       \
  inline v4int operator op( v4int & a, int )   \
  {					       \
    v4int b;                                   \
    _Pragma( "omp simd" )                      \
    for( int j = 0; j < 4; j++ )               \
      b.i[j] = (a.i[j] op);                    \
    return b;                                  \
  }

  POSTFIX_INCDEC(++)
  POSTFIX_INCDEC(--)

# undef POSTFIX_INCDEC

  // v4int binary operators
  
# define BINARY(op)                                             \
  _Pragma( "forceinline recursive" )	                        \
  inline v4int operator op( const v4int &a, const v4int &b )    \
  {								\
    v4int c;                                                    \
    _Pragma( "omp simd" )                                       \
    for( int j = 0; j < 4; j++ )                                \
      c.i[j] = a.i[j] op b.i[j];                                \
    return c;                                                   \
  }

  BINARY(+)
  BINARY(-)
  BINARY(*)
  BINARY(/)
  BINARY(%)
  BINARY(^)
  BINARY(&)
  BINARY(|)
  BINARY(<<)
  BINARY(>>)

# undef BINARY

  // v4int logical operators

# define LOGICAL(op)                                           \
  _Pragma( "forceinline recursive" )	                       \
  inline v4int operator op( const v4int &a, const v4int &b )   \
  {							       \
    v4int c;                                                   \
    _Pragma( "omp simd" )                                      \
    for( int j = 0; j < 4; j++ )                               \
      c.i[j] = -(a.i[j] op b.i[j]);                            \
    return c;                                                  \
  }

  LOGICAL(<)
  LOGICAL(>)
  LOGICAL(==)
  LOGICAL(!=)
  LOGICAL(<=)
  LOGICAL(>=)
  LOGICAL(&&)
  LOGICAL(||)

# undef LOGICAL

  // v4int miscellaneous functions

  #pragma forceinline recursive
  inline v4int abs( const v4int &a )
  {
    v4int b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = ( a.i[j] >= 0 ) ? a.i[j] : -a.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4 czero( const v4int &c, const v4 &a )
  {
    v4 b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = a.i[j] & ~c.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4 notczero( const v4int &c, const v4 &a )
  {
    v4 b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = a.i[j] & c.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4 merge( const v4int &c, const v4 &t, const v4 &f )
  {
    v4 m;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      m.i[j] = ( f.i[j] & ~c.i[j] ) | ( t.i[j] & c.i[j] );

    return m;
  }

  ////////////////
  // v4float class

  class v4float : public v4
  {
    // v4float prefix unary operator friends

    friend inline v4float operator  +( const v4float &a );
    friend inline v4float operator  -( const v4float &a );
    friend inline v4float operator  ~( const v4float &a );
    friend inline v4int   operator  !( const v4float &a );
    // Note: Referencing (*) and dereferencing (&) apply to the whole vector

    // v4float prefix increment / decrement operator friends

    friend inline v4float operator ++( v4float &a );
    friend inline v4float operator --( v4float &a );

    // v4float postfix increment / decrement operator friends

    friend inline v4float operator ++( v4float &a, int );
    friend inline v4float operator --( v4float &a, int );

    // v4float binary operator friends

    friend inline v4float operator  +( const v4float &a, const v4float &b );
    friend inline v4float operator  -( const v4float &a, const v4float &b );
    friend inline v4float operator  *( const v4float &a, const v4float &b );
    friend inline v4float operator  /( const v4float &a, const v4float &b );

    // v4float logical operator friends

    friend inline v4int operator  <( const v4float &a, const v4float &b );
    friend inline v4int operator  >( const v4float &a, const v4float &b );
    friend inline v4int operator ==( const v4float &a, const v4float &b );
    friend inline v4int operator !=( const v4float &a, const v4float &b );
    friend inline v4int operator <=( const v4float &a, const v4float &b );
    friend inline v4int operator >=( const v4float &a, const v4float &b );
    friend inline v4int operator &&( const v4float &a, const v4float &b );
    friend inline v4int operator ||( const v4float &a, const v4float &b );

    // v4float math library friends

#   define CMATH_FR1(fn) friend inline v4float fn( const v4float &a )
#   define CMATH_FR2(fn) friend inline v4float fn( const v4float &a,  \
                                                   const v4float &b )

    CMATH_FR1(acos);  CMATH_FR1(asin);  CMATH_FR1(atan); CMATH_FR2(atan2);
    CMATH_FR1(ceil);  CMATH_FR1(cos);   CMATH_FR1(cosh); CMATH_FR1(exp);
    CMATH_FR1(fabs);  CMATH_FR1(floor); CMATH_FR2(fmod); CMATH_FR1(log);
    CMATH_FR1(log10); CMATH_FR2(pow);   CMATH_FR1(sin);  CMATH_FR1(sinh);
    CMATH_FR1(sqrt);  CMATH_FR1(tan);   CMATH_FR1(tanh);

    CMATH_FR2(copysign);

#   undef CMATH_FR1
#   undef CMATH_FR2

    // v4float miscellaneous friends

    friend inline v4float rsqrt_approx( const v4float &a );
    friend inline v4float rsqrt       ( const v4float &a );
    friend inline v4float rcp_approx( const v4float &a );
    friend inline v4float rcp       ( const v4float &a );
    friend inline v4float fma ( const v4float &a, const v4float &b, const v4float &c );
    friend inline v4float fms ( const v4float &a, const v4float &b, const v4float &c );
    friend inline v4float fnms( const v4float &a, const v4float &b, const v4float &c );
    friend inline v4float  clear_bits( const v4int &m, const v4float &a );
    friend inline v4float    set_bits( const v4int &m, const v4float &a );
    friend inline v4float toggle_bits( const v4int &m, const v4float &a );
    friend inline void increment_4x1( float * ALIGNED(16) p, const v4float &a );
    friend inline void decrement_4x1( float * ALIGNED(16) p, const v4float &a );
    friend inline void     scale_4x1( float * ALIGNED(16) p, const v4float &a );
    // FIXME: crack
    friend inline void trilinear( v4float & wl, v4float & wh );

  public:

    // v4float constructors / destructors

    #pragma forceinline recursive
    v4float() {}                                        // Default constructor

    #pragma forceinline recursive
    v4float( const v4float &a )                         // Copy constructor
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	f[j] = a.f[j];
    }

    #pragma forceinline recursive
    v4float( const v4 &a )                              // Init from mixed
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	f[j] = a.f[j];
    }

    #pragma forceinline recursive
    v4float( float a )                                  // Init from scalar
    {
      #pragma omp simd
      for( int j = 0; j < 4; j++ )
	f[j] = a;
    }

    #pragma forceinline recursive
    v4float( float f0, float f1, float f2, float f3 )   // Init from scalars
    {
      f[0] = f0;
      f[1] = f1;
      f[2] = f2;
      f[3] = f3;
    }

    #pragma forceinline recursive
    ~v4float() {}                                       // Destructor

    // v4float assignment operators

#   define ASSIGN(op)                                   \
    _Pragma( "forceinline recursive" )	                \
    inline v4float &operator op( const v4float &b )     \
    {							\
      _Pragma( "omp simd" )                             \
      for( int j = 0; j < 4; j++ )                      \
        f[j] op b.f[j];		             		\
      return *this;                                     \
    }

    ASSIGN(=)
    ASSIGN(+=)
    ASSIGN(-=)
    ASSIGN(*=)
    ASSIGN(/=)

#   undef ASSIGN

    // v4float member access operator

    #pragma forceinline recursive
    inline float &operator []( int n )
    {
      return f[n];
    }

    #pragma forceinline recursive
    inline float  operator ()( int n )
    {
      return f[n];
    }
  };

  // v4float prefix unary operators

  #pragma forceinline recursive
  inline v4float operator +( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = +a.f[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float operator -( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = -a.f[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4int operator !( const v4float &a )
  {
    v4int b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = a.i[j] ? 0 : -1;

    return b;
  }

  // v4float prefix increment / decrement operators

  #pragma forceinline recursive
  inline v4float operator ++( v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = ++a.f[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float operator --( v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = --a.f[j];

    return b;
  }

  // v4float postfix increment / decrement operators

  #pragma forceinline recursive
  inline v4float operator ++( v4float &a, int )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = a.f[j]++;

    return b;
  }

  #pragma forceinline recursive
  inline v4float operator --( v4float &a, int )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = a.f[j]--;

    return b;
  }

  // v4float binary operators
    
# define BINARY(op)                                                  \
  _Pragma( "forceinline recursive" )	                             \
  inline v4float operator op( const v4float &a, const v4float &b )   \
  {								     \
    v4float c;                                                       \
    _Pragma( "omp simd" )                                            \
    for( int j = 0; j < 4; j++ )                                     \
      c.f[j] = a.f[j] op b.f[j];                                     \
    return c;                                                        \
  }

  BINARY(+)
  BINARY(-)
  BINARY(*)
  BINARY(/)

# undef BINARY

  // v4float logical operators

# define LOGICAL(op)                                               \
  _Pragma( "forceinline recursive" )	                           \
  inline v4int operator op( const v4float &a, const v4float &b )   \
  {								   \
    v4int c;                                                       \
    _Pragma( "omp simd" )                                          \
    for( int j = 0; j < 4; j++ )                                   \
      c.i[j] = - ( a.f[j] op b.f[j] );                             \
    return c;                                                      \
  }

  LOGICAL(< )
  LOGICAL(> )
  LOGICAL(==)
  LOGICAL(!=)
  LOGICAL(<=)
  LOGICAL(>=)
  LOGICAL(&&)
  LOGICAL(||)

# undef LOGICAL

  // v4float math library functions

# define CMATH_FR1(fn)                          \
  _Pragma( "forceinline recursive" )	        \
  inline v4float fn( const v4float &a )         \
  {						\
    v4float b;                                  \
    _Pragma( "omp simd" )                       \
    for( int j = 0; j < 4; j++ )                \
      b.f[j] = ::fn( a.f[j] );                  \
    return b;                                   \
  }

# define CMATH_FR2(fn)                                          \
  _Pragma( "forceinline recursive" )	                        \
  inline v4float fn( const v4float &a, const v4float &b )       \
  {								\
    v4float c;                                                  \
    _Pragma( "omp simd" )                                       \
    for( int j = 0; j < 4; j++ )                                \
      c.f[j] = ::fn( a.f[j], b.f[j] );                          \
    return c;                                                   \
  }

  CMATH_FR1(acos)     CMATH_FR1(asin)  CMATH_FR1(atan) CMATH_FR2(atan2)
  CMATH_FR1(ceil)     CMATH_FR1(cos)   CMATH_FR1(cosh) CMATH_FR1(exp)
  CMATH_FR1(fabs)     CMATH_FR1(floor) CMATH_FR2(fmod) CMATH_FR1(log)
  CMATH_FR1(log10)    CMATH_FR2(pow)   CMATH_FR1(sin)  CMATH_FR1(sinh)
  CMATH_FR1(sqrt)     CMATH_FR1(tan)   CMATH_FR1(tanh)

  #pragma forceinline recursive
  inline v4float copysign( const v4float &a, const v4float &b )
  {
    v4float c;
    float t;
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
    {
      t = ::fabs( a.f[j] );
      if( b.f[j] < 0 ) t = -t;
      c.f[j] = t;
    }
    return c;
  }

# undef CMATH_FR1
# undef CMATH_FR2

  // v4float miscelleanous functions

  #pragma forceinline recursive
  inline v4float rsqrt_approx( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = ::sqrt( 1.0f / a.f[j] );

    return b;
  }

  #pragma forceinline recursive
  inline v4float rsqrt( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = ::sqrt( 1.0f / a.f[j] );

    return b;
  }

  #pragma forceinline recursive
  inline v4float rcp_approx( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = 1.0f / a.f[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float rcp( const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.f[j] = 1.0f / a.f[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float fma( const v4float &a, const v4float &b, const v4float &c )
  {
    v4float d;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      d.f[j] = a.f[j] * b.f[j] + c.f[j];

    return d;
  }

  #pragma forceinline recursive
  inline v4float fms( const v4float &a, const v4float &b, const v4float &c )
  {
    v4float d;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      d.f[j] = a.f[j] * b.f[j] - c.f[j];

    return d;
  }

  #pragma forceinline recursive
  inline v4float fnms( const v4float &a, const v4float &b, const v4float &c )
  {
    v4float d;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      d.f[j] = c.f[j] - a.f[j] * b.f[j];

    return d;
  }

  #pragma forceinline recursive
  inline v4float clear_bits( const v4int &m, const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = ( ~m.i[j] ) & a.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float set_bits( const v4int &m, const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = m.i[j] | a.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline v4float toggle_bits( const v4int &m, const v4float &a )
  {
    v4float b;

    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      b.i[j] = m.i[j] ^ a.i[j];

    return b;
  }

  #pragma forceinline recursive
  inline void increment_4x1( float * ALIGNED(16) p, const v4float &a )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      p[j] += a.f[j];
  }

  #pragma forceinline recursive
  inline void decrement_4x1( float * ALIGNED(16) p, const v4float &a )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      p[j] -= a.f[j];
  }

  #pragma forceinline recursive
  inline void scale_4x1( float * ALIGNED(16) p, const v4float &a )
  {
    #pragma omp simd
    for( int j = 0; j < 4; j++ )
      p[j] *= a.f[j];
  }

  #pragma forceinline recursive
  inline void trilinear( v4float & wl, v4float & wh )
  {
    float x = wl.f[0], y = wl.f[1], z = wl.f[2];

    wl.f[0] = ( ( 1.0f - x ) * ( 1.0f - y ) ) * ( 1.0f - z );
    wl.f[1] = ( ( 1.0f + x ) * ( 1.0f - y ) ) * ( 1.0f - z );
    wl.f[2] = ( ( 1.0f - x ) * ( 1.0f + y ) ) * ( 1.0f - z );
    wl.f[3] = ( ( 1.0f + x ) * ( 1.0f + y ) ) * ( 1.0f - z );

    wh.f[0] = ( ( 1.0f - x ) * ( 1.0f - y ) ) * ( 1.0f + z );
    wh.f[1] = ( ( 1.0f + x ) * ( 1.0f - y ) ) * ( 1.0f + z );
    wh.f[2] = ( ( 1.0f - x ) * ( 1.0f + y ) ) * ( 1.0f + z );
    wh.f[3] = ( ( 1.0f + x ) * ( 1.0f + y ) ) * ( 1.0f + z );
  }

} // namespace v4

#endif // _v4_portable_h_
